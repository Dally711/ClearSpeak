import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from voiceitt_sdk_py.http import TranscribeOptions, VoiceittHttp

from voiceitt_live import build_auth_email, build_auth_user


def create_auth_provider():
    app_id = os.getenv("VOICEITT_APP_ID")
    api_key = os.getenv("VOICEITT_API_KEY")
    login_method = os.getenv("VOICEITT_LOGIN_METHOD", "email")

    if not app_id or not api_key:
        raise RuntimeError("Missing VOICEITT_APP_ID or VOICEITT_API_KEY")

    if login_method == "email":
        email = os.getenv("VOICEITT_EMAIL")
        password = os.getenv("VOICEITT_PASSWORD")
        if not email or not password:
            raise RuntimeError("Email login requires VOICEITT_EMAIL and VOICEITT_PASSWORD")
        return build_auth_email(app_id, api_key, email, password)
    else:
        user_id = os.getenv("VOICEITT_USER_ID")
        return build_auth_user(app_id, api_key, user_id)


def create_app():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    app = Flask(__name__, static_folder="static", static_url_path="")

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/transcribe", methods=["POST"])
    def transcribe():
        if "file" not in request.files:
            return jsonify({"error": "file is required"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "empty filename"}), 400

        # Save upload to a temp file
        suffix = Path(file.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        work_path = tmp_path
        converted_path = None

        # If not already wav, convert to 16k mono WAV via ffmpeg
        if tmp_path.suffix.lower() not in {".wav", ".wave"}:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                tmp_path.unlink(missing_ok=True)
                return jsonify({"error": "ffmpeg is required for live recordings. Install ffmpeg and ensure it is on PATH."}), 400
            converted_path = tmp_path.with_suffix(".converted.wav")
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(tmp_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(converted_path),
            ]
            logging.info("Converting %s to wav via ffmpeg (%s)", tmp_path.name, ffmpeg)
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logging.info("ffmpeg conversion succeeded: %s -> %s", tmp_path.name, converted_path.name)
                work_path = converted_path
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
                logging.exception("ffmpeg conversion failed: %s", stderr)
                tmp_path.unlink(missing_ok=True)
                if converted_path:
                    converted_path.unlink(missing_ok=True)
                return jsonify({"error": "Conversion to WAV failed", "detail": stderr}), 500
            finally:
                tmp_path.unlink(missing_ok=True)

        try:
            auth_provider = create_auth_provider()
            client = VoiceittHttp(auth_provider)
            options = TranscribeOptions(save_audio=False)
            logging.info("Transcribing upload: %s", work_path)
            result = client.transcribe_file(work_path, options)
            text = result.get("text")
            logging.info("Transcript: %s", text)
            return jsonify({"text": text, "raw": result})
        except Exception as exc:
            logging.exception("Transcription failed")
            return jsonify({"error": str(exc)}), 500
        finally:
            for p in (converted_path, work_path):
                if p:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass

    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=False)
