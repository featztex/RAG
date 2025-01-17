import os
import io
import sys
from mistralai import Mistral
from config import api_key

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

key = api_key
model = "mistral-large-latest"
client = Mistral(api_key=key)

test_response = client.chat.complete(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Сколько лет Москве?"
        }
    ]
)

print(test_response.choices[0].message.content)