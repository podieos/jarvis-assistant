# jarvis-assistant

Voice assistants modelled after JARVIS, in two flavors: OpenAI and Google Gemini.

## Layout

```
Google/
  main.py                — Gemini Live realtime assistant (audio + video, tools)
OpenAI/
  GPT-Jarvis/            — Wakeword + Whisper STT + GPT + TTS + camera (most complete)
  GPT-Jarvis Realtime/   — OpenAI Realtime API (websocket, low-latency voice)
  GPT-Text/              — Minimal text-only chat REPL
  GPT-STT/               — Standalone Whisper transcription demo
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit with your keys
```

Set keys in your shell (or via a `.env` loader):

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
```

## Run

Pick one entry point:

```bash
# Google Gemini Live
python Google/main.py

# OpenAI – the full wakeword pipeline
python "OpenAI/GPT-Jarvis/main.py"

# OpenAI Realtime API (websocket)
python "OpenAI/GPT-Jarvis Realtime/main.py"

# OpenAI text-only
python "OpenAI/GPT-Text/main.py"

# OpenAI Whisper transcription of a file
python "OpenAI/GPT-STT/STT.py"
```

## Notes

- The OpenAI wakeword pipeline (`GPT-Jarvis`) needs **SoX** (`rec`/`sox`) installed for recording.
- The OpenAI Realtime version expects a local `hey_jarvis_v0.1.onnx` model — download from [openWakeWord releases](https://github.com/dscripka/openWakeWord/releases) and update `WAKE_MODEL_PATH`.
- The Google version uses [openWakeWord](https://github.com/dscripka/openWakeWord) and OpenCV for the webcam.
- Local files like `log.jsonl`, `memories.jsonl`, `history.json`, recorded `.wav`/`.mp3` and the wakeword `.onnx` are git-ignored.
