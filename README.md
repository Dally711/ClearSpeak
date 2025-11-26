# Voiceitt Live Client (Python)

Small Python client using the official `voiceitt-sdk-py` to:
- Authenticate via app id/api key (speaker-independent) or email/password (personalized).
- Stream microphone audio to Voiceitt over WebSocket (real-time recognition).
- Transcribe a local audio file over HTTP.

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

## Notes
- Do not commit real credentials; use env vars or a private `.env`.
- WebSocket flow: authenticate, create `VoiceittWebsocket`, wait for `on_ready`, stream PCM `float32` chunks.
- If you change commands or personalized models, keep your Voiceitt account configuration in sync before streaming.
