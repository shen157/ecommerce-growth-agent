import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


API_KEY = os.getenv(
    "TOKENHUB_API_KEY"
)

BASE_URL = os.getenv(
    "TOKENHUB_BASE_URL"
)


if not API_KEY:
    raise ValueError(
        "TOKENHUB_API_KEY is missing."
    )

if not BASE_URL:
    raise ValueError(
        "TOKENHUB_BASE_URL is missing."
    )


client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


print("=" * 70)
print("TokenHub Connection Test")
print("=" * 70)


models = client.models.list()


print("\nAvailable Models:")

for model in models.data:
    print(model.id)