"""Parquet caching utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .console import cout

if TYPE_CHECKING:
    from collections.abc import Callable


CACHE_DIR: Path = Path("data")


def _show_cache_stats(df: pd.DataFrame, path: Path) -> None:
    cout(f"  path: [magenta]{path}[/]")
    cout(f"  size: {path.stat().st_size / 1024 / 1024:.1f} MB")
    cout(f"  samples: {len(df):,}")


def parquet_cache(
    path: Path,
    compute: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Load DataFrame from parquet cache, or compute + save it."""

    if path.exists():
        df = pd.read_parquet(path)
        cout("[dim]Loaded from cache:[/]")
        _show_cache_stats(df, path)
        return df

    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    cout("[dim]Cached:[/]")
    _show_cache_stats(df, path)

    return df
