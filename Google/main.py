import asyncio
import json
import traceback
import numpy as np
import pyaudio
from google import genai
from google.genai import types
from openwakeword.model import Model

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# set once in terminal: echo 'export GEMINI_API_KEY="your_key_here"' >> ~/.zshrc && source ~/.zshrc
# set once in terminal: echo 'export OPENWEATHER_API_KEY="your_key_here"' >> ~/.zshrc && source ~/.zshrc
CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
OPENWEATHER_API_KEY = os.environ["OPENWEATHER_API_KEY"]
MODEL = "gemini-3.1-flash-live-preview"
VOICE = "Charon"

WAKE_WORD_MODEL = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
WAKE_WORD_THRESHOLD = 0.6

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
            },
            {
                "name": "end_conversation",
                "description": "End the conversation. Call this when the user says goodbye, thanks you and is done, or clearly has nothing left to discuss.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    }
]


def log_turn(turn):
    with open("log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(turn, ensure_ascii=False) + "\n")


def build_config():
    prompt = """
        You are Jarvis, a sophisticated AI assistant.

        Personality:
        - Address the user as "Sir" occasionally, but naturally
        - Be witty, calm, and slightly dry in humor
        - Keep responses concise — you are a voice assistant, not a chatbot
        - Never use lists or bullet points in speech, speak naturally

        Behavior:
        - For simple questions, be brief — one or two sentences
        - For complex topics, explain clearly but don't ramble
        - If you don't know something, say it

        System:
        - The user is located in Prague, Czech Republic
        - Use Celsius for temperature, metric for distances
        - When using tools, briefly acknowledge it ("Checking the weather, Sir.")
        - When the conversation naturally concludes — such as the user saying goodbye, thanking you, or indicating they're done — call the end_conversation tool
        - Prefer specific tools (get_weather, get_time, etc.) over google_search when they apply
    """
    if os.path.exists("memories.jsonl"):
        with open("memories.jsonl", "r", encoding="utf-8") as f:
            memories = []
            for line in f:
                if line.strip():
                    try:
                        memories.append(json.loads(line))
                    except Exception:
                        pass
        if memories:
            formatted = "\n".join(f"- [{m['timestamp']}] {m['memory']}" for m in memories)
            prompt += f"\n\nMemories from previous conversations:\n{formatted}"

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
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            # NO_INTERRUPTION → model finishes before processing new input (prevents self-hearing)
            # INTERRUPTION    → user can barge in at any time
            activity_handling=types.ActivityHandling.NO_INTERRUPTION,
            automatic_activity_detection=types.AutomaticActivityDetection(
                # HIGH → triggers easily | LOW → needs clear voice, fewer false triggers
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
                # HIGH → cuts off quickly | LOW → waits longer, allows mid-sentence pauses
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=200,    # ms before speech start — increase if first word gets clipped
                silence_duration_ms=1000,  # ms of silence to end turn — increase if model cuts you off
            ),
        ),
    )


def handle_function_call(name, args):
    if name == "get_time":
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elif name == "set_lights":
        return f"Lights turned {args.get('state')}."

    elif name == "get_weather":
        city = args.get("city", "unknown")
        try:
            data = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
                timeout=5
            ).json()

            if data.get("cod") != 200:
                return f"City '{city}' not found."

            return (
                f"Weather in {data['name']}: "
                f"{data['weather'][0]['description']}, "
                f"{data['main']['temp']:.0f}°C, "
                f"feels like {data['main']['feels_like']:.0f}°C, "
                f"humidity {data['main']['humidity']}%, "
                f"wind {data['wind']['speed']} m/s"
            )
        except Exception:
            return f"Failed to fetch weather for {city}."

    elif name == "save_memory":
        memory = args.get("memory", "")
        with open("memories.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "memory": memory}, ensure_ascii=False) + "\n")
        return "Memory saved."

    elif name == "end_conversation":
        return "Goodbye."

    else:
        return f"Unknown function: {name}"


async def wake_word(mic):
    WAKE_WORD_MODEL.reset()
    while True:
        try:
            chunk = await asyncio.to_thread(mic.read, 1280, exception_on_overflow=False)
        except Exception:
            continue
        audio_data = np.frombuffer(chunk, dtype=np.int16)
        predictions = await asyncio.to_thread(WAKE_WORD_MODEL.predict, audio_data)
        for score in predictions.values():
            if score >= WAKE_WORD_THRESHOLD:
                print("Wake word")
                return


async def send_audio(session, mic, stop_event, turn_lock):
    while not stop_event.is_set():
        try:
            chunk = await asyncio.to_thread(mic.read, 640, exception_on_overflow=False)
        except Exception:
            continue
        if turn_lock.locked():
            continue
        try:
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
            )
        except Exception as e:
            print(f"Audio send error: {e}")
            break
    try:
        await session.send_realtime_input(audio_stream_end=True)
    except Exception:
        pass


async def run_tool(session, function, pending_tool_calls, stop_event):
    try:
        result = await asyncio.to_thread(handle_function_call, function.name, function.args or {})
    except asyncio.CancelledError:
        pending_tool_calls.pop(function.id, None)
        return
    if function.id not in pending_tool_calls:
        return
    pending_tool_calls.pop(function.id, None)
    await session.send_tool_response(function_responses=[
        types.FunctionResponse(id=function.id, name=function.name, response={"result": result})
    ])
    if function.name == "end_conversation":
        print("Stop")
        stop_event.set()


async def receive_responses(session, speaker, stop_event, turn_lock):
    global session_handle

    try:
        while True:
            turn = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "input": "", "thought": "", "output": "", "tokens": 0, "modality": {}}
            pending_tool_calls = {}
            responding = False
            print("Listening")
            async for response in session.receive():

                if stop_event.is_set():
                    return

                if response.usage_metadata:
                    usage = response.usage_metadata
                    if usage.total_token_count:
                        turn["tokens"] = usage.total_token_count
                    if usage.response_tokens_details:
                        for detail in usage.response_tokens_details:
                            modality_name = detail.modality.name
                            turn["modality"][modality_name] = turn["modality"].get(modality_name, 0) + detail.token_count

                if response.session_resumption_update:
                    resumption_update = response.session_resumption_update
                    if resumption_update.resumable and resumption_update.new_handle:
                        session_handle = resumption_update.new_handle

                if response.go_away is not None:
                    print("Server requested disconnect, reconnecting...")
                    stop_event.set()
                    return

                if response.tool_call_cancellation:
                    for cancelled_id in response.tool_call_cancellation.ids:
                        task = pending_tool_calls.pop(cancelled_id, None)
                        if task:
                            task.cancel()

                if response.tool_call:
                    for function in response.tool_call.function_calls:
                        task = asyncio.create_task(run_tool(session, function, pending_tool_calls, stop_event))
                        pending_tool_calls[function.id] = task

                content = response.server_content
                if not content:
                    continue

                if content.input_transcription:
                    turn["input"] += content.input_transcription.text

                if content.output_transcription:
                    turn["output"] += content.output_transcription.text
                    if not responding:
                        responding = True
                        print("Responding")

                if content.model_turn:
                    for part in content.model_turn.parts or []:
                        if part.thought:
                            turn["thought"] += part.text
                        elif part.inline_data:
                            async with turn_lock:
                                await asyncio.to_thread(speaker.write, part.inline_data.data)

                if content.generation_complete:
                    print("Completing") # Finishing tools

                if content.turn_complete:
                    log_turn(turn)
                    print("Done")
                    break
    finally:
        if turn.get("input") or turn.get("output"):
            turn["error"] = True
            log_turn(turn)


async def main():
    audio = pyaudio.PyAudio()
    mic = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1280)
    speaker = audio.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    try:
        while True:
            print("Start (wakeword)")
            await wake_word(mic)
            print("Connecting")
            try:
                async with CLIENT.aio.live.connect(model=MODEL, config=build_config()) as session:
                    print("Connected")

                    stop_event = asyncio.Event()
                    turn_lock = asyncio.Lock()
                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(send_audio(session, mic, stop_event, turn_lock))
                        tg.create_task(receive_responses(session, speaker, stop_event, turn_lock))
            except Exception as e:
                print(f"Session error: {e}, reconnecting...")
                    
    except* asyncio.CancelledError:
        pass
    except* Exception as error:
        traceback.print_exception(error)
    finally:
        mic.stop_stream()
        mic.close()
        speaker.stop_stream()
        speaker.close()
        audio.terminate()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Stop")
    pass