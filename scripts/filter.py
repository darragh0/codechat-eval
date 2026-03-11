#!/usr/bin/env python3

"""Dataset filtering (only single-turn conversations & English prompts with Python code)."""

from __future__ import annotations

import re
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache
from os import cpu_count
from threading import Semaphore
from typing import TYPE_CHECKING, Final, cast
from urllib.request import urlretrieve

from datasets import Dataset
from download import load_ds
from fasttext import FastText
from fasttext import load_model as load_fasttext_model
from pandas import DataFrame
from utils.cache import CACHE_DIR, parquet_cache
from utils.console import cout
from utils.display import show_df_overview
from utils.progress import tracked

if TYPE_CHECKING:
    from datasets import Dataset
    from utils.types import DSRow, FilteredDSRow, Uint


LANGS: Final = ("python", "py")
LANG_MODEL_PATH: Final = CACHE_DIR / "lid.176.ftz"


@cache
def get_lang_model() -> FastText._FastText:
    if not LANG_MODEL_PATH.exists():
        LANG_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
            str(LANG_MODEL_PATH),
        )
    return load_fasttext_model(str(LANG_MODEL_PATH))


def extract_md_code_blocks(md: str, /) -> list[str]:
    pattern = rf"```(?:{'|'.join(LANGS)})\n(.*?)```"
    return re.findall(pattern, md, re.DOTALL | re.IGNORECASE)


def is_nontriv_code(code: str, /, *, min_lines: Uint = 5) -> bool:
    lines = [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return len(lines) > min_lines


def is_en_txt(text: str, /) -> bool:
    """Check if text is English using fasttext."""
    model = get_lang_model()
    text = text.replace("\n", " ")
    predictions = model.predict(text)
    return predictions[0][0] == "__label__en"  # type: ignore[reportGeneralTypeIssues]


def process_convo(row: DSRow, /) -> FilteredDSRow | None:
    """Extract first turn if it has English prompt + non-trivial code."""

    conversation = row["conversation"]
    if not conversation or len(conversation[0]) < 2:  # noqa: PLR2004
        return None

    turn = conversation[0]
    user_msg, asst_msg = turn[:2]
    prompt, response = user_msg["content"], asst_msg["content"]

    if user_msg["role"] != "user" or asst_msg["role"] != "assistant" or not prompt or not response:
        return None

    if not is_en_txt(prompt):
        return None

    code_blocks = extract_md_code_blocks(response)
    if not code_blocks or not any(is_nontriv_code(b) for b in code_blocks):
        return None

    return {
        "id": row["conversation_id"],
        "model": row["model"],
        "prompt": prompt,
        "response": response,
        "code": code_blocks,
    }


def filter_rows(ds: Dataset, /) -> DataFrame:
    total = len(ds)
    max_workers = max((cpu_count() or 4) // 2, 1)
    sem = Semaphore(max_workers * 2)

    def _submit(pool: ThreadPoolExecutor, row: object) -> Future[FilteredDSRow | None]:
        sem.acquire()
        return pool.submit(_work, row)

    def _work(row: object) -> FilteredDSRow | None:
        try:
            return process_convo(cast("DSRow", row))
        finally:
            sem.release()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = (_submit(pool, row) for row in ds)  # lazy generator
        records: list[FilteredDSRow] = []
        try:
            for _, future in tracked(futures, "Filtering", total=total):
                result = future.result()
                if result is not None:
                    records.append(result)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    return DataFrame(records)


def filter_ds(ds: Dataset, /) -> DataFrame:
    cache_path = CACHE_DIR / "filtered.parquet"
    cout("Keeping:\n  [bold green]>[/] English prompts & Python code\n  [bold green]>[/] Single-turn conversations\n")

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
