from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pandas import DataFrame, read_parquet

from utils.display import pretty_path

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
            cerr(f"{msg} (removed partial {pretty_path(cache_path)})", exit_code=130, prefix=prefix)
        else:
            cerr(msg, exit_code=130, prefix=prefix)


def _show_cache_stats(df: DataFrame, path: Path) -> None:
    cout(f"  [dim]Location[/]  {pretty_path(path)}")
    cout(f"  [dim]Size[/]      {path.stat().st_size / 1024 / 1024:.1f} MB")
    cout(f"  [dim]Samples[/]   {len(df):,}")


def parquet_cache(
    path: Path,
    compute: Callable[[], DataFrame],
) -> DataFrame:
    """Load DataFrame from parquet cache, or compute + save."""

    if path.exists():
        df = read_parquet(path)
        cout("Loaded from cache:")
        _show_cache_stats(df, path)
        return df

    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)

    cout("Cached:")
    _show_cache_stats(df, path)

    return df
