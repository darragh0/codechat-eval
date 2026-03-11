"""Progress-tracked iteration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .console import cout

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from scripts.utils.types import Uint


def tracked[T](
    items: Iterable[T],
    label: str,
    *,
    total: Uint,
    completed: Uint = 0,
    trans: bool = True,
) -> Iterator[tuple[Uint, T]]:
    """Enumerate items with a rich progress bar + ETA."""

    with Progress(
        SpinnerColumn("flip"),
        TextColumn("{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=cout,
        transient=trans,
    ) as progress:
        task = progress.add_task(label, total=total, completed=completed)
        for i, item in enumerate(items):
            yield i, item
            progress.advance(task)
