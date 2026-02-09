"""Syntactic analysis of code blocks using ruff and radon."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
from radon.complexity import cc_visit
from radon.metrics import mi_visit

from utils.cache import CACHE_DIR, parquet_cache
from utils.console import cerr, cout
from utils.display import section_header, show_df_overview
from utils.progress import tracked
from utils.types import SyntaxEval

if TYPE_CHECKING:
    from utils.types import FilteredDSRow, SyntaxEvalRow


def _check_parseable(blocks: list[str]) -> bool:
    """Check if all code blocks parse via ast.parse."""
    for block in blocks:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                ast.parse(block)
        except SyntaxError:
            return False
    return True


def _run_ruff(code: str) -> dict[str, int]:
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


def _run_radon_complexity(code: str) -> float:
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


def _run_radon_mi(code: str) -> float:
    """Return maintainability index (0-100, higher = more maintainable)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return mi_visit(code, multi=True)
    except Exception:  # noqa: BLE001
        return 0.0


def _analyse_row(code_blocks: list[str]) -> SyntaxEval:
    """Analyse code blocks and return flat score dict."""
    parseable = _check_parseable(code_blocks)
    combined = "\n\n# ===== CODEBLOCK =====\n\n".join(code_blocks)

    ruff_counts = _run_ruff(combined)
    complexity = _run_radon_complexity(combined)
    maintainability = _run_radon_mi(combined)

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


def _print_overview(df: pd.DataFrame) -> None:
    """Print summary statistics for syntax analysis columns."""
    cout("\n[bold]Syntax Analysis Summary:[/]")

    for col in SyntaxEval.__annotations__:
        if col == "parseable":
            n_parseable = df[col].sum()
            cout(f"  {col:<20} {n_parseable:,}/{len(df):,} ({100 * n_parseable / len(df):.1f}%)")
        else:
            mean = df[col].mean()
            med = df[col].median()
            cout(f"  {col:<20} mean={mean:05.2f}  median={med:05.2f}")


def analyse_syntax(
    df: pd.DataFrame,
    /,
    *,
    cache_key: str,
    overview: bool = False,
) -> pd.DataFrame:
    """Run syntactic analysis on each row's code blocks.

    Keyword Args:
        cache_key: Cache key to use (based on filtering args)
    (Optional)
        overview: Show summary statistics after analysis.
    """
    section_header("Syntax Analysis")

    def _compute() -> pd.DataFrame:
        records: list[SyntaxEvalRow] = []
        rows = cast("list[FilteredDSRow]", df.to_dict("records"))

        for _, row in tracked(rows, "Analysing syntax", total=len(rows)):
            code_blocks = row["code"]
            scores = _analyse_row(code_blocks)
            combined = "\n\n# ===== CODEBLOCK =====\n\n".join(code_blocks)

            records.append(
                {
                    "id": row["id"],
                    "model": row["model"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "code": combined,
                    **scores,
                }
            )

        return pd.DataFrame(records)

    cache_path = CACHE_DIR / f"syntax_eval_{cache_key}.parquet"
    result = parquet_cache(cache_path, _compute)

    if overview:
        _print_overview(result)
        show_df_overview(result)

    return result
