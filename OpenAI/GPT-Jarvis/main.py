import os
import subprocess
import numpy as np
import sounddevice as sd
import base64
import time
import json

from pathlib import Path
from openai import OpenAI
from openwakeword.model import Model
from openwakeword import utils

def init():
    global HISTORY_PATH, MAX_HISTORY
    global SR, FRAME_SAMPLES, THRESHOLD, COOLDOWN_S, OPEN_WAKEWORD_MODEL, YES_PATH
    global LAST_WAKE_TS, LAST_COOLDOWN_LOG_TS
    global WAKE_HITS_REQUIRED, WAKE_HIT_COUNT
    global WAKE_LOG_EVERY_N_FRAMES, WAKE_LOG_NEAR_THRESHOLD, WAKE_FRAME_COUNT
    global PHOTO_PATH, VISION_MODEL, VISION_INSTRUCTION, VISION_MAX_TOKENS
    global SOX_TEMP, SOX_OUT, STT_MODEL, STT_LANGUAGE
    global TTS_FILE_PATH, TTS_MODEL, TTS_VOICE, TTS_INSTRUCTIONS, TTS_MIN_CHARS, TTS_MAX_CHARS
    global LLM_MODEL, LLM_MAX_TOKENS, LLM_INSTRUCTIONS
    global CLIENT

    BASE_DIR = Path(__file__).parent
    with open(BASE_DIR / "config.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    HISTORY_PATH = BASE_DIR / "history.json"
    YES_PATH = BASE_DIR / "yes.mp3"
    PHOTO_PATH = BASE_DIR / "photo.jpg"
    SOX_TEMP = BASE_DIR / "temp_44k.wav"
    SOX_OUT = BASE_DIR / "input.wav"
    TTS_FILE_PATH = BASE_DIR / "speech.mp3"

    MAX_HISTORY = data["MAX_HISTORY"]
    SR = data["SR"]
    FRAME_SAMPLES = data["FRAME_SAMPLES"]
    THRESHOLD = data["THRESHOLD"]
    COOLDOWN_S = data["COOLDOWN_S"]
    WAKE_HITS_REQUIRED = data["WAKE_HITS_REQUIRED"]
    WAKE_LOG_EVERY_N_FRAMES = data["WAKE_LOG_EVERY_N_FRAMES"]
    WAKE_LOG_NEAR_THRESHOLD = data["WAKE_LOG_NEAR_THRESHOLD"]

    VISION_MODEL = data["VISION_MODEL"]
    VISION_INSTRUCTION = data["VISION_INSTRUCTION"]
    VISION_MAX_TOKENS = data["VISION_MAX_TOKENS"]

    STT_MODEL = data["STT_MODEL"]
    STT_LANGUAGE = data["STT_LANGUAGE"]

    TTS_MODEL = data.get("TTS_MODEL") or data.get("TTS_GPT_MODEL")
    TTS_VOICE = data.get("TTS_VOICE") or data.get("TTS_GPT_VOICE")
    TTS_INSTRUCTIONS = data.get("TTS_INSTRUCTIONS") or data.get("TTS_GPT_INSTRUCTIONS")
    TTS_MIN_CHARS = data.get("TTS_MIN_CHARS", 0)
    TTS_MAX_CHARS = data.get("TTS_MAX_CHARS", 0)

    LLM_MODEL = data["LLM_MODEL"]
    LLM_MAX_TOKENS = data["LLM_MAX_TOKENS"]
    LLM_INSTRUCTIONS = data["LLM_INSTRUCTIONS"]

    # Prefer env var; fall back to config.json or api_key.txt for local use
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = (data.get("API_KEY") or "").strip()
    if not api_key or api_key == "API_KEY":
        key_file = BASE_DIR / "api_key.txt"
        if key_file.exists():
            api_key = key_file.read_text(encoding="utf-8").strip()
        else:
            raise ValueError("OPENAI_API_KEY not set. Export it or put it in config.json / api_key.txt")
    CLIENT = OpenAI(api_key=api_key)

    utils.download_models()
    OPEN_WAKEWORD_MODEL = Model(wakeword_models=["hey jarvis"], vad_threshold=0.5)

    if not HISTORY_PATH.exists():
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)

    LAST_WAKE_TS = 0.0
    LAST_COOLDOWN_LOG_TS = 0.0
    WAKE_HIT_COUNT = 0
    WAKE_FRAME_COUNT = 0

def delete(p):
    if not p:
        return
    p = Path(p)
    if p.exists():
        p.unlink()

def load_history():
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = data[-MAX_HISTORY:]
    out = [f"user: {i['user']}\nassistant: {i['assistant']}" for i in pairs]
    return "\n".join(out)

def save_history(user, assistant):
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append({"user": str(user).strip(), "assistant": str(assistant).strip()})

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def wake_word():
    global LAST_WAKE_TS, LAST_COOLDOWN_LOG_TS, WAKE_HITS_REQUIRED, WAKE_HIT_COUNT
    global WAKE_LOG_EVERY_N_FRAMES, WAKE_LOG_NEAR_THRESHOLD, WAKE_FRAME_COUNT

    with sd.InputStream(samplerate=SR, channels=1, dtype="int16", blocksize=FRAME_SAMPLES) as stream:
        while True:
            data, _ = stream.read(FRAME_SAMPLES)

            pcm = data[:, 0].astype(np.int16)
            pred = OPEN_WAKEWORD_MODEL.predict(pcm)
            score = max(pred.values()) if isinstance(pred, dict) else float(pred)

            WAKE_FRAME_COUNT += 1
            if (WAKE_FRAME_COUNT % WAKE_LOG_EVERY_N_FRAMES == 0) or (WAKE_LOG_NEAR_THRESHOLD > 0.0 and score >= WAKE_LOG_NEAR_THRESHOLD):
                print(f"SCORE: {score:.2f}")

            if score >= THRESHOLD:
                now = time.time()

                if now - LAST_WAKE_TS < COOLDOWN_S:
                    WAKE_HIT_COUNT = 0
                    if now - LAST_COOLDOWN_LOG_TS >= 1.0:
                        print("Wake detected but cooling down")
                        LAST_COOLDOWN_LOG_TS = now
                    continue

                WAKE_HIT_COUNT += 1
                if WAKE_HIT_COUNT < WAKE_HITS_REQUIRED:
                    continue

                WAKE_HIT_COUNT = 0
                LAST_WAKE_TS = now
                WAKE_FRAME_COUNT = 0
                print("WAKE WORD DETECTED")

                time.sleep(0.25)
                for _ in range(5):
                    stream.read(FRAME_SAMPLES)

                return True
            else:
                WAKE_HIT_COUNT = 0

def record():
    print("START")

    # subprocess.run(["afplay", str(YES_PATH)], check=False) # macOS
    # subprocess.run(["aplay", "-q", str(YES_PATH)], check=False) # Linux
    # subprocess.run(["powershell", "-c", f'(New-Object Media.SoundPlayer "{YES_PATH}").PlaySync();'], check=False) # Windows (WAV only)

    subprocess.run([
        "rec", "-q", "-c", "1", "-r", "44100", "-b", "16", str(SOX_TEMP),
        "silence", "1", "0.1", "2%", "1", "1.5", "2%"
    ], check=True) # Linux/macOS (SoX)

    # subprocess.run([ # Windows (SoX) - requires rec.exe in PATH
    #     "rec", "-q", "-c", "1", "-r", "44100", "-b", "16", str(SOX_TEMP),
    #     "silence", "1", "0.1", "2%", "1", "1.5", "2%"
    # ], check=True)

    subprocess.run([
        "sox", "-q", str(SOX_TEMP), "-r", "16000", "-b", "16", "-c", "1",
        str(SOX_OUT), "dither"
    ], check=True)  # Linux/macOS (SoX)

    # subprocess.run([ # Windows (SoX) - requires sox.exe in PATH
    #     "sox", "-q", str(SOX_TEMP), "-r", "16000", "-b", "16", "-c", "1",
    #     str(SOX_OUT), "dither"
    # ], check=True)

    print("STOP")

def take_photo():
    # subprocess.run(["imagesnap", str(PHOTO_PATH), "-w", "1"], check=True) # macOS
    # subprocess.run([ # Linux (Pi)
    #     "libcamera-still", "-n", "-o", str(PHOTO_PATH),
    #     "--width", "1280", "--height", "720", "--autofocus"
    # ], check=True)
    # subprocess.run(["powershell", "-c", "python win_cam_capture.py"], check=True) # Windows

    raise RuntimeError("Camera capture is disabled (uncomment an OS-specific command).")

def vision(prompt):
    with open(PHOTO_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    vision_resp = CLIENT.responses.create(
        model=VISION_MODEL,
        instructions=VISION_INSTRUCTION,
        max_output_tokens=VISION_MAX_TOKENS,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
            ],
        }],
    )

    out = vision_resp.output_text.strip()
    delete(PHOTO_PATH)
    return out

def stt():
    with open(SOX_OUT, "rb") as audio_file:
        stt_resp = CLIENT.audio.transcriptions.create(
            model=STT_MODEL,
            file=audio_file,
            response_format="text",
            language=STT_LANGUAGE
        )

    text = stt_resp.strip()
    print("YOU:", text)

    delete(SOX_TEMP)
    delete(SOX_OUT)
    return text

def llm(text):
    stream = CLIENT.responses.create(
        model=LLM_MODEL,
        tools=[{"type": "web_search"}],
        tool_choice="auto",
        max_output_tokens=LLM_MAX_TOKENS,
        instructions=LLM_INSTRUCTIONS + load_history(),
        input=text,
        stream=True,
    )

    full = ""
    print("ASSISTANT:", end=" ", flush=True)
    for event in stream:
        if getattr(event, "type", None) != "response.output_text.delta":
            continue
        delta = getattr(event, "delta", "") or ""
        if not delta:
            continue
        print(delta, end="", flush=True)
        full += delta
    print()

    llm_resp = full.strip()

    if "[[CAMERA]]" in llm_resp:
        take_photo()
        cam = vision(text).strip()
        save_history(text, "[[CAMERA]] " + cam)
        print("ASSISTANT (CAMERA):", cam)
        return cam

    save_history(text, llm_resp)
    return llm_resp

def tts(text):
    text = str(text).strip()

    if TTS_MIN_CHARS and len(text) < TTS_MIN_CHARS:
        return
    if TTS_MAX_CHARS and len(text) > TTS_MAX_CHARS:
        text = text[:TTS_MAX_CHARS]

    with CLIENT.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        instructions=TTS_INSTRUCTIONS,
    ) as response:
        response.stream_to_file(TTS_FILE_PATH)

    # from playsound import playsound; playsound(str(TTS_FILE_PATH)) # Windows (pip install playsound)
    # subprocess.run(["afplay", str(TTS_FILE_PATH)], check=False) # macOS
    # subprocess.run(["aplay", str(TTS_FILE_PATH)], check=False) # Linux

    delete(TTS_FILE_PATH)

while True:
    try:
        init()
        wake_word()
        record()
        text = stt()
        resp = llm(text)
        tts(resp)
        delete(SOX_TEMP)
        delete(SOX_OUT)
        delete(PHOTO_PATH)
        delete(TTS_FILE_PATH)
        print("--- NEW CYCLE ---")
    except KeyboardInterrupt:
        print(" --- STOP ---")
        break
    except Exception as e:
        print(f"ERROR: {e}")
        delete(SOX_TEMP)
        delete(SOX_OUT)
        delete(PHOTO_PATH)
        delete(TTS_FILE_PATH)
        print("--- RETRYING ---")
