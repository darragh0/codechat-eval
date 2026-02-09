"""Parquet caching utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .console import cout

if TYPE_CHECKING:
    from collections.abc import Callable


CACHE_DIR: Path = Path("data")


def parquet_cache(
    path: Path,
    compute: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Load DataFrame from parquet cache, or compute + save it."""

    if path.exists():
        df = pd.read_parquet(path)
        cout(f"[dim]Loaded from cache:[/] [magenta]{path}[/] -> {len(df):,} samples")
        return df

    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    cout(f"[dim]Cached:[/] {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")

    return df
