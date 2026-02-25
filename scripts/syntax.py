#!/usr/bin/env python3

"""Syntactic analysis of code blocks using ruff & radon."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from utils.cache import CACHE_DIR, parquet_cache
from utils.console import cerr, cout
from utils.display import show_df_overview
from utils.progress import tracked
from utils.types import SyntaxEval

if TYPE_CHECKING:
    from scripts.utils.types import FilteredDSRow, SyntaxEvalRow


def is_parseable(blocks: list[str]) -> bool:
    """Check if all code blocks parse via ast.parse."""
    for block in blocks:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                ast.parse(block)
        except SyntaxError:
            return False
    return True


def run_ruff(code: str) -> dict[str, int]:
    """Run ruff on code and return violation counts by category."""
    counts = {"errors": 0, "warnings": 0, "flake8": 0, "bugbear": 0, "security": 0}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        tmp = f.name

    ruff = shutil.which("ruff")
    if not ruff:
        cerr("ruff not found in PATH -- see https://docs.astral.sh/ruff/installation/\n")
        emsg = "missing ruff executable"
        raise RuntimeError(emsg)

    try:
        result = subprocess.run(  # noqa: S603
            [
                ruff,
                "check",
                "--isolated",
                "--select",
                "E,W,F,B,S",
                "--output-format=json",
                tmp,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        # ruff exits non-zero when violations found, so check stdout regardless
        if result.stdout.strip():
            violations = json.loads(result.stdout)
            for v in violations:
                rule_code = v.get("code", "")
                if rule_code.startswith("E"):
                    counts["errors"] += 1
                elif rule_code.startswith("W"):
                    counts["warnings"] += 1
                elif rule_code.startswith("F"):
                    counts["flake8"] += 1
                elif rule_code.startswith("B"):
                    counts["bugbear"] += 1
                elif rule_code.startswith("S"):
                    counts["security"] += 1
    finally:
        Path(tmp).unlink(missing_ok=True)

    return counts


def run_radon_complexity(code: str) -> float:
    """Return average cyclomatic complexity (0.0 if no functions/classes)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            blocks = cc_visit(code)
    except Exception:  # noqa: BLE001
        return 0.0
    if not blocks:
        return 0.0
    return sum(b.complexity for b in blocks) / len(blocks)


def run_radon_mi(code: str) -> float:
    """Return maintainability index (0-100, higher = more maintainable)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return mi_visit(code, multi=True)
    except Exception:  # noqa: BLE001
        return 0.0


def analyse_row(code_blocks: list[str]) -> SyntaxEval:
    """Analyse code blocks and return flat score dict."""
    parseable = is_parseable(code_blocks)
    combined = "\n\n# ===== CODEBLOCK =====\n\n".join(code_blocks)

    ruff_counts = run_ruff(combined)
    complexity = run_radon_complexity(combined)
    maintainability = run_radon_mi(combined)

    return {
        "parseable": parseable,
        "lines": sum(block.count("\n") + 1 for block in code_blocks),
        "ruff_errors": ruff_counts["errors"],
        "ruff_warnings": ruff_counts["warnings"],
        "ruff_flake8": ruff_counts["flake8"],
        "ruff_bugbear": ruff_counts["bugbear"],
        "ruff_security": ruff_counts["security"],
        "complexity": complexity,
        "maintainability": maintainability,
    }


def process_syntax_row(row: FilteredDSRow) -> SyntaxEvalRow:
    """Process a single row for syntax analysis."""
    code_blocks = row["code"]
    scores = analyse_row(code_blocks)
    combined = "\n\n# ===== CODEBLOCK =====\n\n".join(code_blocks)

    return {
        "id": row["id"],
        "model": row["model"],
        "prompt": row["prompt"],
        "response": row["response"],
        "code": combined,
        **scores,
    }


def show_oview(df: pd.DataFrame) -> None:
    cout("Syntax Analysis Summary:")

    for col in SyntaxEval.__annotations__:
        if col == "parseable":
            n_parseable = df[col].sum()
            cout(f"  [dim]{col:<20}[/] {n_parseable:,}/{len(df):,} ({100 * n_parseable / len(df):.1f}%)")
        else:
            mean = df[col].mean()
            med = df[col].median()
            cout(f"  [dim]{col:<20}[/] mean={mean:05.2f}  median={med:05.2f}")


def analyse_syntax(df: pd.DataFrame, /) -> pd.DataFrame:
    """Run syntactic analysis on each row's code blocks."""

    def compute() -> pd.DataFrame:
        rows = cast("list[FilteredDSRow]", df.to_dict("records"))
        max_workers = max((os.cpu_count() or 4) // 2, 1)

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(process_syntax_row, row) for row in rows]
            records: list[SyntaxEvalRow] = []
            try:
                for _, fut in tracked(futures, "Analysing syntax", total=len(rows)):
                    records.append(fut.result())
            except KeyboardInterrupt:
                for f in futures:
                    f.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                raise

        records.sort(key=lambda r: r["id"])
        cout()
        return pd.DataFrame(records)

    cache_path = CACHE_DIR / "syntax_eval.parquet"
    result = parquet_cache(cache_path, compute)
    cout()

    show_oview(result)
    cout()
    show_df_overview(result)
    cout()

    return result


def main() -> None:
    filtered_fname = "filtered.parquet"
    cache_path = CACHE_DIR / filtered_fname
    if not cache_path.exists():
        cerr(f"run [cyan]02_filter.py[/] first -- missing [cyan]{filtered_fname}[/]")

    df = pd.read_parquet(cache_path)
    analyse_syntax(df)


if __name__ == "__main__":
    from utils.cache import graceful_exit

    with graceful_exit("syntax analysis stopped"):
        main()
