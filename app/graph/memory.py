from collections.abc import Mapping
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)


def _message_role(
    message: BaseMessage,
) -> str:

    if isinstance(
        message,
        HumanMessage,
    ):
        return "USER"

    if isinstance(
        message,
        AIMessage,
    ):
        return "ASSISTANT"

    return message.type.upper()


MAX_HISTORY_MESSAGES = 6


def build_contextual_query(
    state: Mapping[str, Any],
) -> str:

    query = state["user_query"]

    messages = list(
        state.get(
            "messages",
            [],
        )
    )

    if (
        messages
        and isinstance(
            messages[-1],
            HumanMessage,
        )
        and messages[-1].content == query
    ):
        messages = messages[:-1]

    if not messages:
        return query

    # Keep only recent conversational context.
    messages = messages[
        -MAX_HISTORY_MESSAGES:
    ]

    history = "\n\n".join(
        f"{_message_role(message)}:\n"
        f"{message.content}"
        for message in messages
    )

    return f"""
PREVIOUS CONVERSATION:

{history}


CURRENT USER QUESTION:

{query}
""".strip()