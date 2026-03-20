# ClearSpeak - Live Audio Transcriber for Atypical Speech Patterns (Python)

> The ClearSpeak system is available at the following link (password required): https://clearspeak.onrender.com


## Demo Video

https://github.com/user-attachments/assets/d907efc9-620a-4189-b2bc-ef1525e63eb9


Small Python client using the official **Voiceitt's API** and `voiceitt-sdk-py` to:
- Authenticate via app id/api key (speaker-independent) or email/password (personalized).
- Stream microphone audio to Voiceitt over WebSocket (real-time recognition).
- Transcribe a local audio file over HTTP.
- Serve a lightweight Flask web UI (with a simple password gate) to use the ClearSpeak experience.

## Context
Built for a school project to support a client who survived a traumatic brain injury. The goal is to make communication with others easier and faster by leveraging Voiceitt’s API for personalized speech-to-text. This repo holds a minimal, testable client so we can quickly validate recognition quality for our user.

## Setup
1) Python 3.10+ recommended.
2) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3) Set environment variables (recommended; see `.env.example`):
   - `VOICEITT_APP_ID`
   - `VOICEITT_API_KEY`
   - For personalized: `VOICEITT_EMAIL` and `VOICEITT_PASSWORD`
   - For speaker-independent: optional `VOICEITT_USER_ID`
   - For the web gate: `APP_PASSWORD` (shared password) and `FLASK_SECRET_KEY` (random secret for sessions)

## Usage
Real-time mic (personalized via email/password):
```bash
python voiceitt_live.py --login-method email websocket
```

Real-time mic (app id/api key, speaker-independent):
```bash
python voiceitt_live.py websocket --app-id $VOICEITT_APP_ID --api-key $VOICEITT_API_KEY --user-id $VOICEITT_USER_ID
```

HTTP file transcription (personalized via email/password):
```bash
python voiceitt_live.py --login-method email http --file "/full/path/to/audio.wav"
```

Options:
- `--rate 16000` sample rate (Hz, mono)
- `--chunk-ms 250` send interval in ms
- `--save-audio` ask Voiceitt to save audio server-side (false by default)
- Filenames with spaces are fine; wrap the full path in quotes when using `--file`.
- `--device` microphone device index or name (use `python -c "import sounddevice as sd; import pprint; pprint.pp(sd.query_devices())"` to list; set `PYTHONIOENCODING=utf-8` if your shell chokes on Unicode).

## Web UI (Flask)
- Start the server (uses VOICEITT_* plus APP_PASSWORD/FLASK_SECRET_KEY):
  ```bash
  $env:APP_PASSWORD="your-password"
  $env:FLASK_SECRET_KEY="your-random-64-hex"
  python web_app.py
  ```
- Open http://localhost:5000 in your browser. You’ll see a password screen; on success you reach the ClearSpeak page (`static/index.html`).
- All protected routes (`/`, `/about`, `/team`, `/upload`, `/api/transcribe`) require the session set after login.

## Deploying on Render (quick)
- Build command: `pip install -r requirements.txt`
- Start command: `python web_app.py` (or `gunicorn web_app:app`)
- Environment vars: `APP_PASSWORD`, `FLASK_SECRET_KEY`, `VOICEITT_APP_ID`, `VOICEITT_API_KEY`, `VOICEITT_EMAIL`, `VOICEITT_PASSWORD` (and `VOICEITT_USER_ID` if you switch to user mode). Free tier is fine for a single user; expect cold starts.

## Notes
- Do not commit real credentials; use env vars or a private `.env`.
- WebSocket flow: authenticate, create `VoiceittWebsocket`, wait for `on_ready`, stream PCM `float32` chunks.
- If you change commands or personalized models, keep your Voiceitt account configuration in sync before streaming.
