#!/usr/bin/env python3

"""Dataset filtering (only English prompts with Python code)."""

from __future__ import annotations

import functools
import os
import re
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Semaphore
from typing import TYPE_CHECKING, Final, cast

import fasttext
import pandas as pd
from datasets import Dataset
from download import load_ds
from utils.cache import CACHE_DIR, parquet_cache
from utils.console import cout
from utils.display import show_df_overview
from utils.progress import tracked

if TYPE_CHECKING:
    from datasets import Dataset
    from utils.types import DSRow

    from scripts.utils.types import FilteredDSRow

LANGS: Final = ("python", "py")
LANG_MODEL_PATH: Final = CACHE_DIR / "lid.176.ftz"


@functools.cache
def get_lang_model() -> fasttext.FastText._FastText:
    if not LANG_MODEL_PATH.exists():
        LANG_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
            str(LANG_MODEL_PATH),
        )
    return fasttext.load_model(str(LANG_MODEL_PATH))


def extract_code_blocks(md: str, /) -> list[str]:
    pattern = rf"```(?:{'|'.join(LANGS)})\n(.*?)```"
    return re.findall(pattern, md, re.DOTALL | re.IGNORECASE)


def is_nontriv(code: str, /, *, min_lines: int = 5) -> bool:
    lines = [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return len(lines) > min_lines


def is_en(text: str, /) -> bool:
    """Check if text is English using fasttext."""
    model = get_lang_model()
    text = text.replace("\n", " ")
    predictions = model.predict(text)
    return predictions[0][0] == "__label__en"  # type: ignore[reportGeneralTypeIssues]


def process_conversation(row: DSRow, /) -> list[FilteredDSRow]:
    """Process all turns in a conversation. Returns filtered records."""
    records: list[FilteredDSRow] = []
    prev_turn_id: str | None = None

    for turn_idx, turn in enumerate(row["conversation"]):
        if len(turn) < 2:  # noqa: PLR2004
            break

        user_msg, asst_msg = turn[:2]
        prompt, response = user_msg["content"], asst_msg["content"]

        if user_msg["role"] != "user" or asst_msg["role"] != "assistant" or not prompt or not response:
            continue

        # `user_msg["language"]` exists but is unreliable
        if not is_en(prompt):
            continue

        code_blocks = extract_code_blocks(response)
        if code_blocks and any(is_nontriv(b) for b in code_blocks):
            idee = f"{row['conversation_id']}:{turn_idx}"
            records.append(
                {
                    "id": idee,
                    "model": row["model"],
                    "prompt": prompt,
                    "response": response,
                    "code": code_blocks,
                    "prev_turn_id": prev_turn_id,
                }
            )
            prev_turn_id = idee

    return records


def filter_rows(ds: Dataset, /) -> pd.DataFrame:
    """Filter all rows in the dataset with progress tracking."""
    total = len(ds)
    max_workers = max((os.cpu_count() or 4) // 2, 1)
    extra = (
        f"\n  [bold green]>[/] [dim]English prompts & Python code[/]"
        f"\n  [bold green]>[/] [dim]{max_workers} workers[/]\n"
    )
    sem = Semaphore(max_workers * 2)

    def _submit(pool: ThreadPoolExecutor, row: object) -> Future[list[FilteredDSRow]]:
        sem.acquire()
        return pool.submit(_work, row)

    def _work(row: object) -> list[FilteredDSRow]:
        try:
            return process_conversation(cast("DSRow", row))
        finally:
            sem.release()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = (_submit(pool, row) for row in ds)  # lazy generator
        records: list[FilteredDSRow] = []
        for _, future in tracked(futures, "Filtering", total=total, extra=extra):
            records.extend(future.result())

    df = pd.DataFrame(records)
    cout(f"[bold green]>[/] Filtered to {len(df):,} turns from {len(ds):,} conversations\n")
    return df


def filter_ds(ds: Dataset, /) -> pd.DataFrame:
    """Filter dataset and return as DataFrame. Uses cache if available."""

    cache_path = CACHE_DIR / "filtered.parquet"
    df = parquet_cache(cache_path, lambda: filter_rows(ds))
    cout()

    show_df_overview(df)
    cout()

    return df


def main() -> None:
    filter_ds(load_ds(overview=False))


if __name__ == "__main__":
    from utils.cache import graceful_exit

    with graceful_exit("filtering stopped"):
        main()
