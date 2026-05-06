# Jarvis assistant

Wakeword-triggered voice assistant built on the Gemini Live API and OpenAI Realtime API. Uses openWakeWord for always-on detection, streams bidirectional PCM audio, and supports tool calling (weather, time, memory), webcam video input, and session resumption.

## Layout

```
Google/
  main.py                — Gemini Live realtime assistant (audio + wakeword + tools)
  main (camera).py       — Same as above with live webcam video stream
OpenAI/
  GPT-Jarvis/            — Wakeword + Whisper STT + GPT + TTS + camera (most complete)
  GPT-Jarvis Realtime/   — OpenAI Realtime API (websocket, low-latency voice)
  GPT-Text/              — Minimal text-only chat REPL
  GPT-STT/               — Standalone Whisper transcription demo
```

## Setup

Copy the env file and fill in your keys:

```bash
cp .env.example .env
```

Keys needed:

- `OPENAI_API_KEY` — [platform.openai.com](https://platform.openai.com/api-keys)
- `GEMINI_API_KEY` — [aistudio.google.com](https://aistudio.google.com/apikey)
- `OPENWEATHER_API_KEY` — [openweathermap.org](https://home.openweathermap.org/api_keys) (used by Google versions for weather tool)

Install dependencies for the model you want to run:

```bash
pip install -r Google/requirements.txt
# or
pip install -r "OpenAI/GPT-Jarvis/requirements.txt"
# or
pip install -r "OpenAI/GPT-Jarvis Realtime/requirements.txt"
# or
pip install -r OpenAI/GPT-STT/requirements.txt
# or
pip install -r OpenAI/GPT-Text/requirements.txt
```

## Run

```bash
# Google Gemini Live (audio only)
python Google/main.py

# Google Gemini Live (audio + webcam)
python "Google/main (camera).py"

# OpenAI – full wakeword pipeline
python "OpenAI/GPT-Jarvis/main.py"

# OpenAI Realtime API (websocket)
python "OpenAI/GPT-Jarvis Realtime/main.py"

# OpenAI text-only
python "OpenAI/GPT-Text/main.py"

# OpenAI Whisper transcription
python "OpenAI/GPT-STT/STT.py"
```

## Notes

- The Google versions use [openWakeWord](https://github.com/dscripka/openWakeWord) — place `hey_jarvis_v0.1.onnx` next to the script or update `WAKE_WORD_MODEL`.
- The OpenAI wakeword pipeline (`GPT-Jarvis`) needs **SoX** (`rec`/`sox`) installed for recording.
- The OpenAI Realtime version expects a local `hey_jarvis_v0.1.onnx` model — download from [openWakeWord releases](https://github.com/dscripka/openWakeWord/releases) and update `WAKE_MODEL_PATH`.
- Local files like `log.jsonl`, `memories.jsonl`, `history.json`, recorded `.wav`/`.mp3`, and `.onnx` model files are git-ignored.
