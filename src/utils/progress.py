"""Progress-tracked iteration."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .console import cout
from .display import fmt_eta

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def tracked[T](
    items: Iterable[T],
    label: str,
    *,
    total: int,
    update_every: int = 50,
) -> Iterator[tuple[int, T]]:
    """Enumerate items with a rich status spinner + ETA."""
    start = time.monotonic()

    with cout.status(f"[bold green]{label}...", spinner="flip") as status:
        for i, item in enumerate(items):
            yield i, item
            if i % update_every == 0 and i > 0:
                elapsed = time.monotonic() - start
                rate = i / elapsed
                eta = fmt_eta((total - i) / rate)
                status.update(f"[bold green]{label}... {i:,}/{total:,} [dim]ETA: {eta}[/]")
