import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


TOKENHUB_API_KEY = os.getenv(
    "TOKENHUB_API_KEY"
)

TOKENHUB_BASE_URL = os.getenv(
    "TOKENHUB_BASE_URL"
)

TOKENHUB_MODEL = os.getenv(
    "TOKENHUB_MODEL"
)


def validate_config() -> None:
    """
    Validate TokenHub configuration.
    """

    missing = []

    if not TOKENHUB_API_KEY:
        missing.append(
            "TOKENHUB_API_KEY"
        )

    if not TOKENHUB_BASE_URL:
        missing.append(
            "TOKENHUB_BASE_URL"
        )

    if not TOKENHUB_MODEL:
        missing.append(
            "TOKENHUB_MODEL"
        )

    if missing:
        raise ValueError(
            "Missing environment variables: "
            + ", ".join(missing)
        )


def get_llm(
    temperature: float = 0.0,
):
    """
    Create a LangChain ChatOpenAI client
    connected to Tencent TokenHub.

    TokenHub exposes an OpenAI-compatible API.
    """

    validate_config()

    return ChatOpenAI(
        model=TOKENHUB_MODEL,
        api_key=TOKENHUB_API_KEY,
        base_url=TOKENHUB_BASE_URL,
        temperature=temperature,
    )