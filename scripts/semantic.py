#!/usr/bin/env python3

"""Semantic analysis of prompt-code pairs."""

from __future__ import annotations

from typing import Final

import pandas as pd
from utils.cache import CACHE_DIR, parquet_cache
from utils.console import cerr
from utils.display import show_df_overview

MODEL: Final = "claude-haiku-4-5-20251001"
MAX_TOKENS: Final = 256
SYSTEM_PROMPT: Final = """\
Score a PROMPT/CODE pair across 6 dimensions (1–5 each). Use the full range (3 is genuinely average, not a safe default).
Score correctness against what the PROMPT asked, not an ideal solution.

PROMPT DIMENSIONS

1. CLARITY — How unambiguous is the intent?
   1 Incomprehensible
   2 Mostly unclear — multiple plausible interpretations
   3 Understandable with effort — requires assumptions
   4 Clear to a competent developer
   5 Crystal clear — zero ambiguity

2. SPECIFICITY — How precisely does it describe what is needed?
   1 Completely vague (e.g. "write some Python")
   2 Names a task, no constraints
   3 Core task + some constraints, significant decisions left open
   4 Inputs, outputs, key constraints specified
   5 Fully specified — types, edge cases, examples

3. COMPLETENESS — Enough info to produce a correct answer without guessing?
   1 Missing critical information
   2 Major gaps requiring significant assumptions
   3 Moderate gaps
   4 Nearly complete — minor details inferable
   5 Fully self-contained

CODE DIMENSIONS

4. CORRECTNESS — Does the code solve what was asked?
   1 Wrong or irrelevant
   2 Right idea, critical bugs
   3 Works on happy path, fails common cases
   4 Correct, minor edge-case issues
   5 Fully correct for all cases implied by prompt

5. ROBUSTNESS — Error handling and edge cases?
   1 Fragile — fails on basic inputs
   2 Happy path only
   3 Some defensive coding
   4 Handles most common edge cases
   5 Comprehensive — validation, boundaries, graceful errors

6. READABILITY — Naming, structure, clarity?
   1 Incomprehensible
   2 Poor names, no structure
   3 Acceptable — followable with effort
   4 Good — clear names, logical structure
   5 Exemplary — self-documenting, Pythonic, consistent style

Respond with ONLY valid JSON, no other text:
{"clarity":N,"specificity":N,"completeness":N,"correctness":N,"robustness":N,"readability":N}"""  # noqa: E501, RUF001


def analyse_semantics(df: pd.DataFrame, /) -> pd.DataFrame:
    """Run LLM-as-a-judge semantic analysis on each row after syntax analysis."""

    def compute() -> pd.DataFrame:
        _ = df
        raise NotImplementedError

    cache_path = CACHE_DIR / "semantic_eval.parquet"
    result = parquet_cache(cache_path, compute)

    show_df_overview(result)

    return result


def main() -> None:
    syntax_fname = "syntax_eval.parquet"
    cache_path = CACHE_DIR / syntax_fname
    if not cache_path.exists():
        cerr(f"run [cyan]syntax.py[/] first -- missing [cyan]{syntax_fname}[/]")

    df = pd.read_parquet(cache_path)
    analyse_semantics(df)


if __name__ == "__main__":
    from utils.cache import graceful_exit

    with graceful_exit():
        main()
