from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .console import cout

if TYPE_CHECKING:
    from pandas import DataFrame


def pretty_path(path: Path, /) -> str:
    return f"[magenta]{f'{path.resolve()}'.replace(str(Path.home()), '~', 1)}[/]"


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
