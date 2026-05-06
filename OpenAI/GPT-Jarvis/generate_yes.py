import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="ballad",
    input="Yes.",
) as r:
    r.stream_to_file("yes.mp3")
