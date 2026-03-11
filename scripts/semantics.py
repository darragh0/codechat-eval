#!/usr/bin/env python3

"""Semantic analysis of prompt-code pairs via Ollama LLM."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final, Literal, cast, get_args

from ollama import Client
from pandas import DataFrame, read_parquet
from utils.cache import CACHE_DIR, parquet_cache
from utils.console import cerr, cout
from utils.display import show_df_overview
from utils.progress import tracked
from utils.types import CodeSemEval, PromptSemEval, Uint

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from utils.types import SemanticEvalRow, SyntaxEvalRow


type Dim = Literal["clarity", "specificity", "completeness", "correctness", "robustness", "readability", "efficiency"]

DIMS: Final[set[Dim]] = set(get_args(Dim.__value__))
MODEL: Final = "qwen3-coder:30b"
MAX_SINGLE_RETRY: Final[Uint] = 10

SYSTEM_PROMPT: Final = """SYSTEM_PROMPT:
Task: Score a PROMPT-CODE pair across 7 quality dimensions (1-5 each).

You will be provided with the following inputs:
  0. INSTRUCTIONS: These are referencial instructions about the strict JSON output requirement for your response.
  1. USER_PROMPT: This is a user's prompt wherein they request assistance with programming-related problems.
  2. LLM_RESPONSE: This is the code an LLM has generated to solve the user's problem (as requested in USER_PROMPT).
  3. LLM_CODE: This is a concatenated string containing all code blocks from the LLM's response (LLM_RESPONSE).

The inputs are provided in the user message. Inputs can be found by their XLM-like open and closing tags.
For example, the "INSTRUCTIONS" inputs is encapsulated within <INSTRUCTIONS> and </INSTRUCTIONS>.

Use the full range (3 means genuinely average, not a safe default).
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

Reason briefly about each dimension, then output STRICT JSON in the following format. \
This is the ONLY text that should be in your final output. \
No introduction, no discussion, ONLY THIS JSON (EXCLUDING backticks and "json" tag).

You must NEVER deviate from this response format:
```json
{"clarity":N,"specificity":N,"completeness":N,"correctness":N,"robustness":N,"readability":N,"efficiency":N}
```"""


def json_find_and_loads(txt: str) -> dict:
    """Find, extract & load JSON from str."""

    snip = f"{txt[: -(len(txt) / 2)]!r}"
    end = txt.rfind("}")
    if end == -1:
        msg = f"[JSON] No close brace (`}}`): ... {snip}"
        raise ValueError(msg)

    start = txt.rfind("{", 0, end)
    if start == -1:
        msg = f"[JSON] No open brace (`{{`): ... {snip}"
        raise ValueError(msg)

    try:
        js = json.loads(txt[start : end + 1])
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON: {snip}"
        raise ValueError(msg) from e

    return js


def check_json_fmt(js: dict) -> None:
    """Check JSON vs. required format (also clamp int overflows in-place)."""
    missing_keys = set(DIMS) - js.keys()
    if missing_keys:
        msg = f"Missing keys: {sorted(missing_keys)!r}"
        raise ValueError(msg)

    errkeys: set[str] = set()
    for k, v in js.items():
        if not isinstance(v, int):
            errkeys.add(k)
        js[k] = max(1, min(5, v))

    if errkeys:
        msg = f"Non-integer values for keys: {sorted(errkeys)!r}"
        raise TypeError(msg)


def get_llm_json(txt: str | None) -> dict:
    """Extract last JSON object from CoT (Chain of Thought) response."""
    if txt is None:
        msg = "LLM returned None"
        raise ValueError(msg)

    js = json_find_and_loads(txt)
    check_json_fmt(js)
    return js


def check_ollama(model: str) -> Client:
    """Verify Ollama running & model available."""
    try:
        client = Client()
        available = client.list().models
    except Exception as e:
        cerr("Ollama not running -- start it with [cyan]ollama serve[/]", exit_code=1)
        raise RuntimeError("unreachable") from e

    has_model = any(False if m.model is None else m.model.startswith(model) for m in available)
    if not has_model:
        cerr(f"model [cyan]{model}[/] not found -- pull it with [cyan]ollama pull {model}[/]", exit_code=1)

    return client


def score_row(client: Client, row: SyntaxEvalRow) -> dict:
    """Call LLM & extract JSON scores."""

    meow = (
        """<INSTRUCTIONS>Reason briefly about each dimension as per the system instructions. \
Output STRICT JSON in the following format. This is the ONLY text that should be \
in your final output — no introduction, no discussion, ONLY THIS JSON (EXCLUDING backticks and "json" tag).

You must NEVER deviate from this response format:
```json
{"clarity":N,"specificity":N,"completeness":N,"correctness":N,"robustness":N,"readability":N,"efficiency":N}
```</INSTRUCTIONS>"""
        f"<USER_PROMPT>{row['prompt']}</USER_PROMPT>"
        f"<LLM_RESPONSE>:{row['response']}</LLM_RESPONSE>"
        f"<LLM_CODE>:{row['code']}</LLM_CODE>"
    )

    resp = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": meow},
        ],
    )

    cout(meow)
    cout("\n-----\n")
    cout(resp)
    return get_llm_json(resp.message.content)


def process_row(client: Client, row: SyntaxEvalRow) -> SemanticEvalRow:
    """Evaluate single row on all prompt + code dimensions."""

    err: TypeError | ValueError | None = None
    for _ in range(MAX_SINGLE_RETRY):
        try:
            raw = score_row(client, row)
            break
        except (TypeError, ValueError) as e:
            err = e
    else:
        cerr(f"could not score row {row['id']} after {MAX_SINGLE_RETRY} retries", exit_code=1)
        raise err if err else RuntimeError("unreachable")

    return cast("SemanticEvalRow", {**row, **{dim: raw[dim] for dim in DIMS}})


def load_checkpoint(path: Path) -> list[SemanticEvalRow]:
    """Load completed rows from JSONL checkpoint file."""
    if not path.exists():
        return []

    records: list[SemanticEvalRow] = []
    with path.open() as f:
        records.extend([cast("SemanticEvalRow", json.loads(line.strip())) for line in f if line.strip()])
    return records


@contextmanager
def checkpoint_writer(path: Path) -> Generator[Callable[[SemanticEvalRow], None]]:
    """Yield function to append scored rows to checkpoint file."""
    with path.open("a") as f:

        def write(row: SemanticEvalRow) -> None:
            f.write(json.dumps(row) + "\n")
            f.flush()

        yield write


def show_oview(df: DataFrame) -> None:
    cout("Semantic Analysis Summary:")

    dims = [*PromptSemEval.__annotations__, *CodeSemEval.__annotations__]
    for col in dims:
        mean = df[col].mean()
        med = df[col].median()
        cout(f"  [dim]{col:<20}[/] mean={mean:.2f}  median={med:.2f}")


def analyse_semantics(df: DataFrame) -> DataFrame:
    """Run LLM-as-a-judge semantic analysis on each row."""

    cache_path = CACHE_DIR / "semantic_eval.parquet"
    checkpoint_path = CACHE_DIR / "semantic_eval.checkpoint.jsonl"

    def compute() -> DataFrame:
        client = check_ollama(MODEL)
        all_rows = cast("list[SyntaxEvalRow]", df.to_dict("records"))

        done = load_checkpoint(checkpoint_path)
        done_ids = {r["id"] for r in done}
        remaining = [r for r in all_rows if r["id"] not in done_ids]

        with checkpoint_writer(checkpoint_path) as write:
            for _, row in tracked(remaining, "Scoring semantics", total=len(all_rows), completed=len(done)):
                result = process_row(client, row)
                done.append(result)
                write(result)

        done.sort(key=lambda r: r["id"])
        checkpoint_path.unlink(missing_ok=True)
        return DataFrame(done)

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
        cerr(f"run [cyan]scripts/syntax.py[/] first -- missing [cyan]{syntax_fname}[/]", exit_code=1)

    df = read_parquet(cache_path)
    analyse_semantics(df)


if __name__ == "__main__":
    from utils.cache import graceful_exit

    with graceful_exit("semantic analysis stopped"):
        main()
