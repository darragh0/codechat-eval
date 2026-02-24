"""Parquet caching utilities."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd
from utils.display import prettypath

from .console import cerr, cout

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


CACHE_DIR: Final = Path(__file__).parents[2] / "data"


@contextmanager
def graceful_exit(msg: str = "exiting", *, cache_path: Path | None = None) -> Generator[None]:
    """Catch KeyboardInterrupt, clean up partial cache file, & exit."""
    try:
        yield
    except KeyboardInterrupt:
        prefix = "[bold yellow]>[/]"
        if cache_path and cache_path.exists():
            cache_path.unlink()
            cerr(f"{msg} (removed partial {prettypath(cache_path)})", exit_code=130, prefix=prefix)
        else:
            cerr(msg, exit_code=130, prefix=prefix)


def _show_cache_stats(df: pd.DataFrame, path: Path) -> None:
    cout(f"  [dim]Location[/]  {prettypath(path)}")
    cout(f"  [dim]Size[/]      {path.stat().st_size / 1024 / 1024:.1f} MB")
    cout(f"  [dim]Samples[/]   {len(df):,}")


def parquet_cache(
    path: Path,
    compute: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Load DataFrame from parquet cache, or compute + save it."""

    if path.exists():
        df = pd.read_parquet(path)
        cout("Loaded from cache:")
        _show_cache_stats(df, path)
        return df

    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

    cout("Cached:")
    _show_cache_stats(df, path)

    return df
