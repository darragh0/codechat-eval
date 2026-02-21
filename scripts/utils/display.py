from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .console import cout

if TYPE_CHECKING:
    from collections.abc import Iterable, Sized

    from pandas import DataFrame


def prettypath(path: Path, /) -> str:
    return f"[magenta]{f'{path.resolve()}'.replace(str(Path.home()), '~', 1)}[/]"


def maxlen(items: Iterable[Sized], /) -> int:
    """Return length of longest item in an iterable of sized objects."""
    return max(len(x) for x in items)


def show_df_overview(df: DataFrame, /) -> None:
    """Show DataFrame overview."""
    cout("DataFrame Info")
    cout(f"  [dim]Shape[/]   {df.shape}")
    cout(f"  [dim]Models[/]  {df['model'].nunique()} unique")
    cout("  [dim]Columns[/]")

    col_width = max(len(c) for c in df.columns)
    zpad = len(str(df.shape[1]))

    for i, col in enumerate(df.columns):
        cout(f"    {i:0{zpad}}  {col:<{col_width}}\t[cyan]{df[col].dtype}[/]")


def fmt_eta(seconds: float) -> str:
    """Format seconds as human-readable time remaining."""

    if seconds < 60:  # noqa: PLR2004
        return f"{seconds:.0f}s"
    mins, secs = divmod(int(seconds), 60)

    if mins < 60:  # noqa: PLR2004
        return f"{mins}m {secs:02d}s"
    hrs, mins = divmod(mins, 60)

    return f"{hrs}h {mins:02d}m"
