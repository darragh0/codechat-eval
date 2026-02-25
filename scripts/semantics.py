#!/usr/bin/env python3

"""Semantic analysis of prompt-code pairs via local LLM (Ollama)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, cast

import ollama
import pandas as pd
from utils.cache import CACHE_DIR, parquet_cache
from utils.console import cerr, cout
from utils.display import show_df_overview
from utils.progress import tracked
from utils.types import CodeSemEval, PromptSemEval

if TYPE_CHECKING:
    from pathlib import Path

    from utils.types import SemanticEvalRow, SyntaxEvalRow

MODEL: Final = "qwen3-coder:30b"
DIMENSIONS: Final = ("clarity", "specificity", "completeness", "correctness", "robustness", "readability", "efficiency")

SYSTEM_PROMPT: Final = """\
Score a PROMPT/CODE pair across 7 dimensions (1-5 each). \
Use the full range (3 is genuinely average, not a safe default).
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

7. EFFICIENCY — Is the algorithmic approach appropriate?
   1 Fundamentally wrong approach (exponential where linear exists)
   2 Naive brute-force, orders of magnitude slower than needed
   3 Reasonable approach, not optimal
   4 Good algorithmic choices, minor optimisation possible
   5 Optimal or near-optimal approach for the problem

Reason briefly about each dimension, then output JSON on its own line:
{"clarity":N,"specificity":N,"completeness":N,"correctness":N,"robustness":N,"readability":N,"efficiency":N}"""


def extract_json(text: str | None) -> dict:
    """Extract last JSON object from CoT response."""

    if text is None:
        raise ValueError("LLM returned None response")

    end = text.rfind("}")
    if end == -1:
        raise ValueError(f"No JSON found in response: {text[:200]!r}")

    start = text.rfind("{", 0, end)
    if start == -1:
        raise ValueError(f"No JSON found in response: {text[:200]!r}")

    return json.loads(text[start : end + 1])


def clamp_score(val: object) -> int:
    """Clamp value to the 1-5 score range."""
    return max(1, min(5, int(val)))  # type: ignore[arg-type]


def check_ollama(model: str) -> None:
    """Verify Ollama is running and the model is available."""
    try:
        client = ollama.Client()
        available = client.list().models
    except Exception:  # noqa: BLE001
        cerr("Ollama is not running -- start it with [cyan]ollama serve[/]", exit_code=1)
        return  # unreachable

    if not any(False if m.model is None else m.model.startswith(model) for m in available):
        cerr(f"model [cyan]{model}[/] not found -- pull it with [cyan]ollama pull {model}[/]", exit_code=1)


def score_row(client: ollama.Client, row: SyntaxEvalRow) -> dict:
    """Call the LLM and extract the JSON scores dict."""
    resp = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"PROMPT:\n{row['prompt']}\n\nCODE:\n{row['code']}"},
        ],
    )

    cout(resp)
    return extract_json(resp.message.content)


def process_row(client: ollama.Client, row: SyntaxEvalRow) -> SemanticEvalRow:
    """Evaluate a single row on all prompt + code dimensions."""
    try:
        raw = score_row(client, row)
        missing = set(DIMENSIONS) - raw.keys()
    except ValueError:
        raw = None
        missing = None

    if raw is None or missing:
        try:
            raw = score_row(client, row)
        except ValueError:
            raise
        missing = set(DIMENSIONS) - raw.keys()
        if missing:
            msg = f"Row {row['id']}: missing {sorted(missing)} after retry. Got: {raw}"
            raise ValueError(msg)

    return cast(
        "SemanticEvalRow",
        {
            **row,
            **{dim: clamp_score(raw[dim]) for dim in DIMENSIONS},
        },
    )


def load_checkpoint(path: Path) -> list[SemanticEvalRow]:
    """Load completed rows from a JSONL checkpoint file."""
    if not path.exists():
        return []

    records: list[SemanticEvalRow] = []
    with path.open() as f:
        records.extend([cast("SemanticEvalRow", json.loads(line.strip())) for line in f if line.strip()])
    return records


def append_checkpoint(path: Path, row: SemanticEvalRow) -> None:
    """Append a single scored row to the checkpoint file."""
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def show_oview(df: pd.DataFrame) -> None:
    cout("Semantic Analysis Summary:")

    dims = [*PromptSemEval.__annotations__, *CodeSemEval.__annotations__]
    for col in dims:
        mean = df[col].mean()
        med = df[col].median()
        cout(f"  [dim]{col:<20}[/] mean={mean:.2f}  median={med:.2f}")


def analyse_semantics(df: pd.DataFrame) -> pd.DataFrame:
    """Run LLM-as-a-judge semantic analysis on each row."""

    cache_path = CACHE_DIR / "semantic_eval.parquet"
    checkpoint_path = CACHE_DIR / "semantic_eval.checkpoint.jsonl"

    def compute() -> pd.DataFrame:
        check_ollama(MODEL)
        client = ollama.Client()
        all_rows = cast("list[SyntaxEvalRow]", df.to_dict("records"))

        done = load_checkpoint(checkpoint_path)
        done_ids = {r["id"] for r in done}
        remaining = [r for r in all_rows if r["id"] not in done_ids]

        for _, row in tracked(remaining, "Scoring semantics", total=len(all_rows), completed=len(done)):
            result = process_row(client, row)
            done.append(result)
            append_checkpoint(checkpoint_path, result)

        done.sort(key=lambda r: r["id"])
        checkpoint_path.unlink(missing_ok=True)
        return pd.DataFrame(done)

    result = parquet_cache(cache_path, compute)
    cout()

    show_oview(result)
    cout()
    show_df_overview(result)
    cout()

    return result


def main() -> None:
    syntax_fname = "syntax_eval.parquet"
    cache_path = CACHE_DIR / syntax_fname
    if not cache_path.exists():
        cerr(f"run [cyan]03_syntax.py[/] first -- missing [cyan]{syntax_fname}[/]")

    df = pd.read_parquet(cache_path)
    analyse_semantics(df)


if __name__ == "__main__":
    from utils.cache import graceful_exit

    with graceful_exit("semantic analysis stopped"):
        main()
