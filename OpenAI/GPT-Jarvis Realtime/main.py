import base64
import json
import os
import queue
import ssl
import threading
import time

import certifi
import numpy as np
import sounddevice as sd
import websocket

from openwakeword.model import Model

WAKE_MODEL_PATH = "/path/to/hey_jarvis_v0.1.onnx"  # download from github.com/dscripka/openWakeWord/releases
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini-realtime-preview"
VOICE = "echo"
INSTRUCTIONS = (
    "You are JARVIS: calm, precise, efficient. Default language: English. Short answers. "
    "When you decide the conversation should end, include the word 'bye' in your final sentence."
)

WAKE_SR = 16000
WAKE_BLOCK = 1280
WAKE_THRESHOLD = 0.6

def send(ws, obj):
    ws.send(json.dumps(obj))

def b64e(b):
    return base64.b64encode(b).decode("ascii")

def b64d(s):
    return base64.b64decode(s)

def log(s):
    print(s, flush=True)

speaker = sd.RawOutputStream(samplerate=24000, channels=1, dtype="int16")
speaker.start()

q = None
mic_stream = None

IDLE, USER, ASSIST = range(3)
state = IDLE
mic_muted = False
mute_until = 0.0
last_done_time = 0.0
assistant_text = ""

def sender_loop(ws):
    while True:
        pcm = q.get()
        if pcm is None:
            return
        if mic_muted:
            continue
        try:
            send(ws, {"type": "input_audio_buffer.append", "audio": b64e(pcm)})
        except Exception:
            return

def start_mic():
    global mic_stream
    def cb(indata, frames, time_info, status):
        if status:
            return
        if mic_muted:
            return
        if time.time() < mute_until:
            return
        try:
            q.put_nowait(indata.tobytes())
        except queue.Full:
            pass

    mic_stream = sd.InputStream(
        samplerate=24000,
        channels=1,
        dtype=np.int16,
        blocksize=int(24000 * 0.02),
        callback=cb,
    )
    mic_stream.start()

def stop_mic():
    global mic_stream
    if mic_stream is None:
        return
    try:
        mic_stream.stop()
    except Exception:
        pass
    try:
        mic_stream.close()
    except Exception:
        pass
    mic_stream = None

def on_open(ws):
    global state, mic_muted, mute_until, last_done_time, assistant_text
    state = IDLE
    mic_muted = False
    mute_until = 0.0
    last_done_time = 0.0
    assistant_text = ""

    send(ws, {
        "type": "session.update",
        "session": {
            "modalities": ["audio", "text"],
            "voice": VOICE,
            "instructions": INSTRUCTIONS,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.85,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 700,
                "create_response": True,
                "interrupt_response": False,
            },
        },
    })

    threading.Thread(target=sender_loop, args=(ws,), daemon=True).start()
    start_mic()

def on_message(ws, msg):
    global state, mic_muted, mute_until, last_done_time, assistant_text
    e = json.loads(msg)
    t = e.get("type")
    now = time.time()

    if t == "input_audio_buffer.speech_started":
        if state == IDLE and not mic_muted and now >= mute_until:
            state = USER
            log("talking")
        return

    if t == "input_audio_buffer.speech_stopped":
        if state == USER:
            log("response")
            state = IDLE
        return

    if t in (
        "response.text.delta",
        "response.output_text.delta",
        "response.audio_transcript.delta",
        "response.output_audio_transcript.delta",
    ):
        assistant_text += (e.get("delta") or "")
        return

    if t == "response.audio.delta":
        state = ASSIST
        mic_muted = True
        speaker.write(b64d(e.get("delta", "")))
        return

    if t == "response.done":
        if (now - last_done_time) < 0.9:
            return
        last_done_time = now
        state = IDLE
        mic_muted = False
        mute_until = now + 0.6

        if "bye" in assistant_text.lower():
            assistant_text = ""
            log("STOP")
            try:
                ws.close()
            except Exception:
                pass
            return

        assistant_text = ""
        log("done")
        return

    if t == "error":
        log(f"error: {e}")

def on_error(ws, err):
    log(f"error: {err}")

def on_close(ws, code, reason):
    log(f"closed: {code} {reason}")
    stop_mic()
    try:
        q.put_nowait(None)
    except Exception:
        pass

def wake_word():
    while True:
        try:
            with sd.InputStream(samplerate=WAKE_SR, channels=1, dtype="int16", blocksize=WAKE_BLOCK) as stream:
                while True:
                    audio, _ = stream.read(WAKE_BLOCK)
                    pcm = audio[:, 0].astype(np.int16)
                    pred = wake_model.predict(pcm)
                    score = max(pred.values()) if isinstance(pred, dict) else float(pred)
                    if score >= WAKE_THRESHOLD:
                        print("WAKE", flush=True)
                        return
        except sd.PortAudioError:
            time.sleep(0.8)

while True:
    wake_model = Model(
        wakeword_model_paths=[WAKE_MODEL_PATH],
        vad_threshold=0.5,
    )
    wake_word()

    q = queue.Queue(maxsize=800)

    ws = websocket.WebSocketApp(
        f"wss://api.openai.com/v1/realtime?model={MODEL}",
        header=[f"Authorization: Bearer {OPENAI_API_KEY}", "OpenAI-Beta: realtime=v1"],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws.run_forever(
        sslopt={"cert_reqs": ssl.CERT_REQUIRED, "ca_certs": certifi.where()},
        ping_interval=20,
        ping_timeout=10,
    )

    time.sleep(0.8)
