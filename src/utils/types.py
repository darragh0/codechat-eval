from datetime import datetime as dt
from typing import Annotated, Literal, TypedDict

from annotated_types import Ge

Uint = Annotated[int, Ge(0)]
ChatRole = Literal["user", "assistant"]
OutputMethod = Literal["raw", "pretty", "both"]


class _ConversationMsg(TypedDict):
    """Single message in conversation."""

    role: ChatRole
    content: str
    language: str
    timestamp: dt


class DSRow(TypedDict):
    """Original dataset row."""

    conversation_id: str
    model: str
    conversation: list[list[_ConversationMsg]]
    turn: int
    snippet_turns: list[int]


class FilteredDSRow(TypedDict):
    """Filtered dataset row."""

    id: str
    model: str
    prompt: str
    response: str
    code: list[str]
