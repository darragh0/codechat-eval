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


def tracked[T](
    items: Iterable[T],
    label: str,
    *,
    total: int,
    completed: int = 0,
    extra: str = "",
) -> Iterator[tuple[int, T]]:
    """Enumerate items with a rich progress bar + ETA."""
    if extra:
        cout(extra.strip())

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[dim]{task.description}[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=cout,
    ) as progress:
        task = progress.add_task(label, total=total, completed=completed)
        for i, item in enumerate(items):
            yield i, item
            progress.advance(task)
