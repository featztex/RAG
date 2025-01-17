import os
from mistralai import Mistral
from config import api_key

key = api_key
model = "mistral-large-latest"
client = Mistral(api_key=key)

test_response = client.chat.complete(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of Romania?"
        }
    ]
)

print(test_response.choices[0].message.content)
