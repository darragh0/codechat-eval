"""Dataset filtering."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, cast

import pandas as pd
from datasets import Dataset
from langdetect import LangDetectException, detect

from utils.cache import CACHE_DIR, parquet_cache
from utils.console import cout
from utils.display import section_header, show_df_overview
from utils.progress import tracked
from utils.types import DSRow

if TYPE_CHECKING:
    from collections.abc import Iterator

    from datasets import Dataset

    from utils.types import FilteredDSRow

LANGS = ("python", "py")


def _rows_as[T](ds: Dataset, _: type[T], /) -> Iterator[T]:
    """Cast dataset rows to type `T`."""
    for r in ds:
        yield cast("T", r)


def _extract_code_blocks(md: str, /) -> list[str]:
    """Extract all code blocks (for Python) from markdown."""
    pattern = rf"```(?:{'|'.join(LANGS)})\n(.*?)```"
    return re.findall(pattern, md, re.DOTALL | re.IGNORECASE)


def _is_nontrivial(code: str, /, *, min_lines: int = 5) -> bool:
    """Check if code has enough meaningful lines."""
    lines = [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return len(lines) > min_lines


def _is_english(text: str, /) -> bool:
    """Check if text is English using language detection."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False  # Skip on undetectable


def _process_row(row: DSRow, /) -> FilteredDSRow | None:
    """Process a single row. Returns filtered record or None."""
    conversation = row["conversation"]
    if not conversation or len(conversation[0]) < 2:  # noqa: PLR2004
        return None

    first_turn = conversation[0]
    user_msg, asst_msg = first_turn[:2]
    prompt, response = user_msg["content"], asst_msg["content"]

    if user_msg["role"] != "user" or asst_msg["role"] != "assistant" or not prompt or not response:
        return None

    if user_msg["language"] != "English" or not _is_english(prompt):
        return None

    code_blocks = _extract_code_blocks(response)
    if not code_blocks:
        return None

    if not any(_is_nontrivial(b) for b in code_blocks):
        return None

    return {
        "id": row["conversation_id"],
        "model": row["model"],
        "prompt": prompt,
        "response": response,
        "code": code_blocks,
    }


def _filter_rows(ds: Dataset, /) -> pd.DataFrame:
    """Filter all rows in the dataset with progress tracking."""
    records: list[FilteredDSRow] = []

    for _, row in tracked(_rows_as(ds, DSRow), "Filtering", total=len(ds), update_every=100):
        if record := _process_row(row):
            records.append(record)

    return pd.DataFrame(records)


def filter_ds(ds: Dataset, /, *, overview: bool = False) -> pd.DataFrame:
    """Filter dataset and return as DataFrame. Uses cache if available."""
    section_header("Filtering")

    cout("Filtering single-turn English conversations with Python code\n")

    cache_path = CACHE_DIR / "filtered.parquet"

    df = parquet_cache(cache_path, lambda: _filter_rows(ds))

    cout(f"[dim]Filtered:[/] {len(df):,}/{len(ds):,} conversations ({100 * len(df) / len(ds):.1f}%)")
    if overview:
        show_df_overview(df)

    return df
