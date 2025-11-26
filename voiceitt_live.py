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
):
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()
    ready_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if status:
            logging.warning("Audio status: %s", status)
        audio_queue.put(indata.copy())

    blocksize = int(rate * (chunk_ms / 1000))
    stream = sd.InputStream(
        samplerate=rate,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        callback=audio_callback,
    )

    def on_ready_changed(is_ready: bool):
        logging.info("Model ready: %s", is_ready)
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
        logging.info("Starting to send microphone audio")
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            ws.send_pcm_audio(chunk.flatten(), "float32")
            time.sleep(chunk_ms / 1000)

    sender_thread = threading.Thread(target=sender, daemon=True)

    def shutdown(*_):
        stop_event.set()
        ws.close()
        stream.stop()
        stream.close()
        logging.info("Stopped")

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logging.info("Connecting to Voiceitt WebSocket")
    options = VoiceittWebsocketOptions(
        save_audio=save_audio,
        show_debug_logs=show_debug_logs,
        handle_sigint=False,  # let our handler manage Ctrl+C
    )
    try:
        ws.connect(options)
    except Exception as exc:  # pragma: no cover - live connection path
        logging.error("WebSocket connect failed: %s", exc)
        return
    sender_thread.start()

    logging.info("Opening microphone (rate=%s, chunk=%sms)", rate, chunk_ms)
    try:
        with stream:
            ws.wait()
    except KeyboardInterrupt:
        shutdown()


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
