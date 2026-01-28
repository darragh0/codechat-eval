"""Dataset filtering."""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from datasets import Dataset
from langdetect import LangDetectException, detect
from rich.rule import Rule

from utils.console import cout
from utils.ds import rows_as, show_ds_overview
from utils.types import DSRow

if TYPE_CHECKING:
    from typing import Literal

    from utils.types import FilteredDSRow

CACHE_DIR: Final = Path("data")


def _cache_key(*, only_english: bool, langs: tuple[str, ...] | Literal["*"]) -> str:
    """Generate a short hash key from filter arguments."""
    canonical = f"{only_english}|{langs}"
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _extract_code_blocks(md: str, /, langs: tuple[str, ...] | Literal["*"]) -> list[str]:
    """Extract all code blocks (for the given language) from markdown."""
    pattern = r"```\w*\n(.*?)```" if langs == "*" else rf"```(?:{'|'.join(langs)})\n(.*?)```"
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


def _process_row(row: DSRow, /, *, only_english: bool, langs: tuple[str, ...] | Literal["*"]) -> FilteredDSRow | None:
    """Process a single row. Returns filtered record or None."""
    # Get first turn (user + assistant message pair)
    conversation = row["conversation"]
    if not conversation or len(conversation[0]) < 2:  # noqa: PLR2004
        return None

    first_turn = conversation[0]
    user_msg, asst_msg = first_turn[:2]
    prompt, response = user_msg["content"], asst_msg["content"]

    if user_msg["role"] != "user" or asst_msg["role"] != "assistant" or not prompt or not response:
        return None

    if only_english and (
        user_msg["language"] != "English"  # Metadata pre-filter
        or not _is_english(prompt)  # Verify with actual detection
    ):
        return None

    # Extract non-trivial code blocks
    code_blocks = _extract_code_blocks(response, langs=langs)
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


def _format_eta(seconds: float) -> str:
    """Format seconds as human-readable time remaining."""
    if seconds < 60:  # noqa: PLR2004
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:  # noqa: PLR2004
        return f"{minutes}m {secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m"


def _filter_rows(
    ds: Dataset,
    /,
    *,
    only_english: bool,
    langs: tuple[str, ...] | Literal["*"],
) -> list[FilteredDSRow]:
    """Filter all rows in the dataset with progress tracking."""
    records: list[FilteredDSRow] = []
    total = len(ds)
    start_time = time.monotonic()
    status_update_interval = 100

    with cout.status("[bold green]Filtering...", spinner="flip") as status:
        for i, row in enumerate(rows_as(ds, DSRow)):
            if record := _process_row(row, only_english=only_english, langs=langs):
                records.append(record)

            if i % status_update_interval == 0 and i > 0:
                elapsed = time.monotonic() - start_time
                rate = i / elapsed
                remaining = (total - i) / rate
                eta = _format_eta(remaining)
                status.update(f"[bold green]Filtering... {i:,}/{total:,} ({len(records):,} kept) [dim]ETA: {eta}[/]")

    return records


def filter_ds(
    ds: Dataset,
    /,
    *,
    only_english: bool,
    langs: tuple[str, ...] | Literal["*"],
    overview: bool = False,
) -> Dataset:
    """Filter dataset and return as Dataset. Uses cache if available.

    Keyword Args:
        only_english: Filter out non-English conversations
        langs: Programming languages to extract code blocks for ("*" for all)
    (Optional)
        overview: Show dataset overview
    """

    cout(Rule("[dim]Filtering[/]", style="dim"))
    cout()

    cache_key = _cache_key(only_english=only_english, langs=langs)
    cache_path = CACHE_DIR / f"filtered_{cache_key}.parquet"

    if cache_path.exists():
        loaded = Dataset.from_parquet(str(cache_path))
        assert isinstance(loaded, Dataset)
        filtered_ds = loaded
        cout(f"[dim]Loaded from cache:[/] {len(filtered_ds):,} samples")

    else:
        records = _filter_rows(ds, only_english=only_english, langs=langs)
        filtered_ds = Dataset.from_list(cast("list[dict]", records))

        # Save cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        filtered_ds.to_parquet(cache_path)
        cout(f"[dim]Cached:[/] {cache_path} ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)\n")

    cout(f"[dim]Filtered:[/] {len(filtered_ds):,}/{len(ds):,} conversations ({100 * len(filtered_ds) / len(ds):.1f}%)")

    if overview:
        from ds import DS_NAME  # noqa: PLC0415

        show_ds_overview(filtered_ds, ds_name=f"{DS_NAME} (filtered)")

    return filtered_ds
