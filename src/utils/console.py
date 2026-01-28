"""Console printing (shorthands for `cout.print` and `cerr.print`)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

from rich.console import Console

if TYPE_CHECKING:
    from rich.console import JustifyMethod, OverflowMethod
    from rich.style import Style


class RichConsolePrintKwargs(TypedDict, total=False):
    """Typed keyword arguments for `rich.console.print`."""

    sep: str
    end: str
    style: str | Style | None
    justify: JustifyMethod | None
    overflow: OverflowMethod | None
    no_wrap: bool | None
    emoji: bool | None
    markup: bool | None
    highlight: bool | None
    width: int | None
    height: int | None
    crop: bool
    soft_wrap: bool | None
    new_line_start: bool


class _Console(Console):
    def __call__(
        self,
        *objects: Any,  # noqa: ANN401
        **kwargs: Unpack[RichConsolePrintKwargs],
    ) -> None:
        self.print(*objects, **kwargs)


class _ErrConsole(Console):
    def __call__(
        self,
        *objects: Any,  # noqa: ANN401
        exit_code: int | None = None,
        **kwargs: Unpack[RichConsolePrintKwargs],
    ) -> None:
        self.print("[bold red]error:[/]", *objects, **kwargs)
        if exit_code is not None:
            sys.exit(exit_code)


cout = _Console()
cerr = _ErrConsole(stderr=True)
