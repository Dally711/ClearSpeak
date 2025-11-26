import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd

from voiceitt_sdk_py.voiceitt_auth_provider import VoiceittApi, VoiceittAuthProvider
from voiceitt_sdk_py.http import TranscribeOptions, VoiceittHttp
from voiceitt_sdk_py.websocket import VoiceittWebsocket, VoiceittWebsocketOptions


def build_auth_user(app_id: str, api_key: str, user_id: str | None) -> VoiceittAuthProvider:
    api = VoiceittApi()
    data = api.sign_in(app_id, api_key, user_id or "")
    if "token" not in data or "refresh_token" not in data:
        raise RuntimeError(f"Sign-in failed: {data}")
    return VoiceittAuthProvider(
        data["token"],
        data["refresh_token"],
        data["token_expires_at"],
        data["refresh_token_expires_at"],
    )


def build_auth_email(app_id: str, api_key: str, email: str, password: str) -> VoiceittAuthProvider:
    url = "https://api2.voiceitt.com/v1/auth/login/email"
    payload = {
        "app_id": app_id,
        "api_key": api_key,
        "email": email,
        "password": password,
    }
    resp = httpx.post(url, json=payload, timeout=20.0)
    data = resp.json()
    if resp.status_code != 200 or "token" not in data or "refresh_token" not in data:
        raise RuntimeError(f"Email sign-in failed ({resp.status_code}): {data}")
    data["token_expires_at"] = data.get("token_expires_at", 0) / 1000
    data["refresh_token_expires_at"] = data.get("refresh_token_expires_at", 0) / 1000
    return VoiceittAuthProvider(
        data["token"],
        data["refresh_token"],
        data["token_expires_at"],
        data["refresh_token_expires_at"],
    )


def stream_microphone_ws(
    auth_provider: VoiceittAuthProvider,
    rate: int,
    chunk_ms: int,
    save_audio: bool,
    show_debug_logs: bool,
    device: str | None,
):
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()
    ready_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if status:
            logging.warning("Audio status: %s", status)
        audio_queue.put(indata.copy())

    blocksize = int(rate * (chunk_ms / 1000))
    # Allow selecting a specific input device by index or name
    sd_device = None
    if device is not None:
        try:
            sd_device = int(device)
        except ValueError:
            sd_device = device

    logging.debug("Opening input stream (device=%s, rate=%s, chunk_ms=%s)", sd_device, rate, chunk_ms)
    try:
        stream = sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=audio_callback,
            device=sd_device,
        )
    except Exception as exc:  # pragma: no cover - live device path
        logging.error("Failed to open microphone (device=%s, rate=%s): %s", sd_device, rate, exc)
        return

    def on_ready_changed(is_ready: bool):
        logging.debug("Model ready: %s", is_ready)
        if is_ready:
            ready_event.set()
        else:
            ready_event.clear()

    def on_partial_recognition(data):
        text = data.get("text") or data.get("unstable_text")
        if text:
            logging.info("Partial: %s", text)

    def on_recognition(data):
        text = data.get("text")
        if text:
            logging.info("Final: %s", text)

    def on_error(data):
        logging.error("Error: %s", data)

    ws = VoiceittWebsocket(
        auth_provider,
        on_ready_changed=on_ready_changed,
        on_partial_recognition=on_partial_recognition,
        on_recognition=on_recognition,
        on_error=on_error,
    )

    def sender():
        ready_event.wait()
        logging.debug("Starting to send microphone audio")
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            ws.send_pcm_audio(chunk.flatten(), "float32")
            time.sleep(chunk_ms / 1000)

    sender_thread = threading.Thread(target=sender, daemon=True)

    def shutdown(*_):
        if stop_event.is_set():
            return
        stop_event.set()
        try:
            ws.close()
        except Exception:
            pass
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        logging.debug("Stopped")

    logging.debug("Connecting to Voiceitt WebSocket")
    options = VoiceittWebsocketOptions(
        save_audio=save_audio,
        show_debug_logs=show_debug_logs,
        handle_sigint=False,
    )
    try:
        ws.connect(options)
    except Exception as exc:  # pragma: no cover - live connection path
        logging.error("WebSocket connect failed: %s", exc)
        return
    sender_thread.start()

    # Run the websocket wait loop in its own thread so Ctrl+C can interrupt the main thread.
    ws_thread = threading.Thread(target=ws.wait, daemon=True)
    ws_thread.start()

    logging.debug("Opening microphone (rate=%s, chunk=%sms)", rate, chunk_ms)
    try:
        with stream:
            while ws_thread.is_alive():
                time.sleep(0.1)
    except KeyboardInterrupt:
        shutdown()
    finally:
        shutdown()
        ws_thread.join(timeout=1.0)
        sender_thread.join(timeout=1.0)


def transcribe_file_http(
    auth_provider: VoiceittAuthProvider, file_path: Path, save_audio: bool
):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if isinstance(file_path, str):
        file_path = Path(file_path)
    client = VoiceittHttp(auth_provider)
    options = TranscribeOptions(save_audio=save_audio)
    logging.info("Sending file to Voiceitt: %s", file_path)
    result = client.transcribe_file(file_path, options)
    logging.info("Transcription result: %s", result)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Voiceitt real-time microphone or HTTP file transcription using voiceitt-sdk-py",
    )
    parser.add_argument("--app-id", default=os.getenv("VOICEITT_APP_ID"), required=False)
    parser.add_argument(
        "--api-key", default=os.getenv("VOICEITT_API_KEY"), required=False
    )
    parser.add_argument(
        "--login-method",
        choices=["user_id", "email"],
        default=os.getenv("VOICEITT_LOGIN_METHOD", "user_id"),
        help="user_id (default) or email login",
    )
    parser.add_argument(
        "--user-id", default=os.getenv("VOICEITT_USER_ID"), required=False
    )
    parser.add_argument(
        "--email", default=os.getenv("VOICEITT_EMAIL"), required=False
    )
    parser.add_argument(
        "--password", default=os.getenv("VOICEITT_PASSWORD"), required=False
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    ws_parser = subparsers.add_parser("websocket", help="Stream microphone audio")
    ws_parser.add_argument("--rate", type=int, default=16000, help="Sample rate (Hz)")
    ws_parser.add_argument(
        "--chunk-ms", type=int, default=250, help="Chunk duration in milliseconds"
    )
    ws_parser.add_argument(
        "--save-audio", action="store_true", help="Ask Voiceitt to save audio"
    )
    ws_parser.add_argument(
        "--ws-debug",
        action="store_true",
        help="Enable verbose WebSocket logs from the SDK",
    )
    ws_parser.add_argument(
        "--device",
        help="Input device index or name for microphone (see sounddevice.query_devices)",
    )

    http_parser = subparsers.add_parser("http", help="Transcribe a file over HTTP")
    http_parser.add_argument("--file", required=True, help="Path to audio file")
    http_parser.add_argument(
        "--save-audio", action="store_true", help="Ask Voiceitt to save audio"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if args.log_level != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("socketio").setLevel(logging.WARNING)
        logging.getLogger("engineio").setLevel(logging.WARNING)

    missing = [name for name, val in [("app-id", args.app_id), ("api-key", args.api_key)] if not val]
    if missing:
        logging.error("Missing required credentials: %s", ", ".join(missing))
        sys.exit(1)

    try:
        if args.login_method == "email":
            if not args.email or not args.password:
                logging.error("Email login requires --email and --password (or VOICEITT_EMAIL/VOICEITT_PASSWORD).")
                sys.exit(1)
            auth_provider = build_auth_email(args.app_id, args.api_key, args.email, args.password)
        else:
            auth_provider = build_auth_user(args.app_id, args.api_key, args.user_id)
    except Exception as exc:
        logging.error("Authentication failed: %s", exc)
        sys.exit(1)

    if args.mode == "websocket":
        stream_microphone_ws(
            auth_provider=auth_provider,
            rate=args.rate,
            chunk_ms=args.chunk_ms,
            save_audio=args.save_audio,
            show_debug_logs=args.ws_debug,
            device=args.device,
        )
    elif args.mode == "http":
        transcribe_file_http(
            auth_provider=auth_provider,
            file_path=Path(args.file),
            save_audio=args.save_audio,
        )
    else:
        logging.error("Unknown mode: %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
