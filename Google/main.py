import asyncio
import json
import traceback
import numpy as np
import pyaudio
from google import genai
from google.genai import types
from openwakeword.model import Model

import os
import cv2
import requests
from datetime import datetime

# set once in terminal: echo 'export GEMINI_API_KEY="your_key_here"' >> ~/.bashrc && source ~/.bashrc
CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.1-flash-live-preview"
VOICE = "Charon"

WAKE_WORD_MODEL = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
WAKE_WORD_THRESHOLD = 0.5

total_tokens = 0
session_handle = None

tools = [
    {"google_search": {}},
    {
        "function_declarations": [
            {
                "name": "set_lights",
                "description": "Control the lights. Use when the user wants more or less light.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "state": {"type": "string", "description": "on or off"}
                    },
                    "required": ["state"]
                }
            },
            {
                "name": "get_weather",
                "description": "Get current weather. Use when the user needs weather or temperature info.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name."}
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "get_time",
                "description": "Get the current date and time. Use when the user asks what time or date it is.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "save_memory",
                "description": "Save something to long-term memory. Use when anything is worth remembering for future conversations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory": {"type": "string", "description": "A short one-line summary of what to remember."}
                    },
                    "required": ["memory"]
                }
            }
        ]
    }
]


def log_turn(turn: dict):
    with open("log.jsonl", "a") as f:
        f.write(json.dumps(turn) + "\n")


def build_config():
    prompt = f"""You are Jarvis, a sophisticated AI assistant inspired by Jarvis from Iron Man.
        Personality:
        - Address the user as "Sir" at all times
        - Be witty, calm, and slightly dry in humor — like the original Jarvis
        - Keep responses concise — you are a voice assistant, not a chatbot
        - Never use lists or bullet points in speech, speak naturally

        Context:
        - The user is located in Prague, Czech Republic
        - Use Celsius for temperature, metric for distances
        Behavior:
        - For simple questions, be brief — one or two sentences
        - For complex topics, explain clearly but don't ramble
        - When using tools, briefly acknowledge what you're doing ("Checking the weather, Sir.")
        - If you don't know something, say so honestly with Jarvis-like charm

        When the conversation naturally concludes — such as the user saying goodbye, thanking you, indicating they're done, or there's nothing left to discuss — respond with [DONE] in your message.
    """
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=prompt,
        tools=tools,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
            )
        ),
        # "minimal" → fastest | "low" → better facts | "medium" → complex | "high" → deepest, slowest
        thinking_config=types.ThinkingConfig(
            thinking_level="minimal",
        ),
        # passes saved handle on reconnect so model remembers full history — valid for 2hr
        session_resumption=types.SessionResumptionConfig(
            handle=session_handle  # None = fresh session
        ),
        # MEDIA_RESOLUTION_LOW → cheaper | MEDIA_RESOLUTION_MEDIUM → default | MEDIA_RESOLUTION_HIGH → fine text
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        # compresses old history so session never dies from the 15min/2min context limit
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(),
        ),
        realtime_input_config=types.RealtimeInputConfig(
            # TURN_INCLUDES_ONLY_ACTIVITY → video only while speaking (~$2/mo) | TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO → continuous vision (~$11/mo)
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            # NO_INTERRUPTION → model finishes before processing new input (prevents self-hearing)
            # INTERRUPTION    → user can barge in at any time
            activity_handling=types.ActivityHandling.NO_INTERRUPTION,
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,  # True = push-to-talk, send activityStart/activityEnd manually
                # HIGH → triggers easily | LOW → needs clear voice, fewer false triggers
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                # HIGH → cuts off quickly | LOW → waits longer, allows mid-sentence pauses
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=200,    # ms before speech start — increase if first word gets clipped
                silence_duration_ms=1000,  # ms of silence to end turn — increase if model cuts you off
            ),
        ),
    )


def handle_function_call(name: str, args: dict) -> str:
    if name == "get_time":
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elif name == "set_lights":
        return f"Lights turned {args.get('state')}."

    elif name == "get_weather":
        city = args.get("city", "unknown")
        try:
            location = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1}, timeout=5).json()

            if not location.get("results"):
                return f"City '{city}' not found."

            lat = location["results"][0]["latitude"]
            lon = location["results"][0]["longitude"]
            city_name = location["results"][0]["name"]

            weather = requests.get("https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                }, timeout=5).json()

            current = weather["current"]
            return (
                f"Weather in {city_name}: "
                f"{current['temperature_2m']}°C, "
                f"humidity {current['relative_humidity_2m']}%, "
                f"wind {current['wind_speed_10m']} km/h, "
                f"code {current['weather_code']}"
            )
        except Exception:
            return f"Failed to fetch weather for {city}."

    elif name == "save_memory":
        memory = args.get("memory", "")
        with open("memories.jsonl", "a") as f:
            f.write(json.dumps({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "memory": memory}) + "\n")
        return "Memory saved."

    else:
        return f"Unknown function: {name}"


async def wake_word():
    WAKE_WORD_MODEL.reset()
    while True:
        chunk = await asyncio.to_thread(MICROPHONE.read, 1280, exception_on_overflow=False)
        audio_data = np.frombuffer(chunk, dtype=np.int16)
        predictions = await asyncio.to_thread(WAKE_WORD_MODEL.predict, audio_data)
        for score in predictions.values():
            if score >= WAKE_WORD_THRESHOLD:
                print("Wake word")
                return


async def send_audio(session, stop_event: asyncio.Event, turn_lock: asyncio.Lock):
    while not stop_event.is_set():
        chunk = await asyncio.to_thread(MICROPHONE.read, 640, exception_on_overflow=False)
        if not turn_lock.locked():
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
            )
    await session.send_realtime_input(audio_stream_end=True)


async def receive_responses(session, stop_event: asyncio.Event, turn_lock: asyncio.Lock, interrupted_event: asyncio.Event):
    global total_tokens, session_handle

    while True:
        turn = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "input": "", "thought": "", "output": "", "tokens": 0, "modality": {}}
        state = "Listening"
        pending_tool_calls = set()
        print("Listening")
        async for response in session.receive():

            if response.usage_metadata:
                usage = response.usage_metadata
                if usage.total_token_count:
                    total_tokens += usage.total_token_count
                    turn["tokens"] = total_tokens
                if usage.response_tokens_details:
                    for d in usage.response_tokens_details:
                        turn["modality"][d.modality.name] = turn["modality"].get(d.modality.name, 0) + d.token_count

            if response.session_resumption_update:
                resumption_update = response.session_resumption_update
                if resumption_update.resumable and resumption_update.new_handle:
                    session_handle = resumption_update.new_handle

            if response.go_away is not None:
                print(f"Connection closing in {response.go_away.time_left}")

            if response.tool_call_cancellation:
                for cancelled_id in response.tool_call_cancellation.ids:
                    pending_tool_calls.discard(cancelled_id)

            if response.tool_call:
                for function in response.tool_call.function_calls:
                    pending_tool_calls.add(function.id)
                function_responses = []
                for function in response.tool_call.function_calls:
                    result = await asyncio.to_thread(handle_function_call, function.name, function.args)
                    if function.id in pending_tool_calls:
                        function_responses.append(types.FunctionResponse(
                            id=function.id,
                            name=function.name,
                            response={"result": result}
                        ))
                        pending_tool_calls.discard(function.id)
                if function_responses:
                    await session.send_tool_response(function_responses=function_responses)

            content = response.server_content
            if not content:
                continue

            if content.interrupted is True:
                interrupted_event.set()
                print("Interrupted")

            if content.input_transcription:
                turn["input"] += content.input_transcription.text

            if content.output_transcription:
                turn["output"] += content.output_transcription.text
                if state != "Responding":
                    state = "Responding"
                    print("Responding")

            if content.model_turn:
                for part in content.model_turn.parts:
                    if part.thought:
                        turn["thought"] += part.text
                    elif part.inline_data:
                        if not interrupted_event.is_set():
                            if state != "Responding":
                                state = "Responding"
                                print("Responding")
                            async with turn_lock:
                                await asyncio.to_thread(SPEAKER.write, part.inline_data.data)

            if content.generation_complete:
                print("Completing") # Finishing tools

            if content.turn_complete:
                interrupted_event.clear()
                log_turn(turn)
                print("Done")
                if "[DONE]" in turn["output"]:
                    print("Stop")
                    stop_event.set()
                    return
                break


async def send_video(session, stop_event: asyncio.Event):
    cap = cv2.VideoCapture(0)
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                _, jpeg = cv2.imencode(".jpg", frame)
                await session.send_realtime_input(
                    video=types.Blob(data=jpeg.tobytes(), mime_type="image/jpeg")
                )
            await asyncio.sleep(1)  # max 1fps as per API limit
    finally:
        cap.release()


async def main():
    global MICROPHONE, SPEAKER
    audio = pyaudio.PyAudio()
    MICROPHONE = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)
    SPEAKER    = audio.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    try:
        while True:
            print("Start (wakeword)")
            await wake_word()
            print("Connecting")
            async with CLIENT.aio.live.connect(model=MODEL, config=build_config()) as session:
                print("Connected")

                if os.path.exists("memories.jsonl"):
                    with open("memories.jsonl", "r") as f:
                        memories = [json.loads(line) for line in f if line.strip()]
                    if memories:
                        formatted = "\n".join(f"- [{m['timestamp']}] {m['memory']}" for m in memories)
                        await session.send_client_content(
                            turns=[{"role": "user", "parts": [{"text": f"Memories from previous conversations:\n{formatted}"}]}],
                            turn_complete=False
                        )

                stop_event = asyncio.Event()
                turn_lock  = asyncio.Lock()
                interrupted_event = asyncio.Event()
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(send_audio(session, stop_event, turn_lock))
                    tg.create_task(send_video(session, stop_event))
                    tg.create_task(receive_responses(session, stop_event, turn_lock, interrupted_event))

    except* asyncio.CancelledError:
        pass
    except* Exception as error:
        MICROPHONE.close()
        SPEAKER.close()
        traceback.print_exception(error)
    finally:
        audio.terminate()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Stop")
    pass
