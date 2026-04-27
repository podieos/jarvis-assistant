import os
from openai import OpenAI

CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

while True:
    question = input("Ask: ").strip()

    response = CLIENT.responses.create(
        model="gpt-4o-mini",
        input=question,
        max_output_tokens=200,
    )

    print(response.output_text.strip())
