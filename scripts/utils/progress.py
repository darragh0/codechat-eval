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
    eta_update_every: int = 1_000,
    extra: str = "",
) -> Iterator[tuple[int, T]]:
    """Enumerate items with a rich status spinner + ETA."""
    start = time.monotonic()

    with cout.status(f"[dim]{label}{extra}[/]", spinner="flip") as status:
        last_eta_str: str | None = None
        last_eta_time = 0.0
        for i, item in enumerate(items):
            yield i, item
            if i % update_every == 0 and i > 0:
                elapsed = time.monotonic() - start
                rate = i / elapsed

                if last_eta_str is None or (elapsed - last_eta_time) * 1000 >= eta_update_every:
                    last_eta_str = fmt_eta((total - i) / rate)
                    last_eta_time = elapsed

                update = f"[dim]{label}[/] [cyan][{i:,}/{total:,}][/] [green][ETA: {last_eta_str}][/]"
                status.update(f"{update}{extra}")
