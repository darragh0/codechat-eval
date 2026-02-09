from datetime import datetime as dt
from typing import Annotated, Literal, TypedDict

from annotated_types import Ge, Le

Uint = Annotated[int, Ge(0)]
Ufloat = Annotated[float, Ge(0.0)]
ChatRole = Literal["user", "assistant"]
Percent = Annotated[float, Ge(0.0), Le(100.0)]
PromptSemScore = Annotated[int, Ge(1), Le(5)]
CodeSemScore = PromptSemScore


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
    turn: Uint
    snippet_turns: list[int]


class FilteredDSRow(TypedDict):
    """Filtered dataset row."""

    id: str
    model: str
    prompt: str
    response: str
    code: list[str]


class SyntaxEval(TypedDict):
    parseable: bool  # all individual blocks parse via ast.parse
    lines: Uint  # total lines of code (all blocks combined)
    ruff_errors: Uint  # E* (style errors)
    ruff_warnings: Uint  # W* (style warnings)
    ruff_flake8: Uint  # F* (logical issues)
    ruff_bugbear: Uint  # B* (bug patterns)
    ruff_security: Uint  # S* (security issues)
    complexity: Annotated[float, Ge(1)]  # radon
    maintainability: Percent  # radon


class SyntaxEvalRow(SyntaxEval):
    """Row with syntactic evaluation scores."""

    id: str
    model: str
    prompt: str
    response: str
    code: str  # all blocks concatenated


class PromptSemEval(TypedDict):
    clarity: PromptSemScore
    specificity: PromptSemScore
    completeness: PromptSemScore


class CodeSemEval(TypedDict):
    correctness: CodeSemScore
    robustness: CodeSemScore
    readability: CodeSemScore
