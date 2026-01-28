"""Dataset loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from datasets import load_dataset
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule

from utils.console import cout
from utils.ds import show_ds_overview

if TYPE_CHECKING:
    from datasets import Dataset

    from utils.types import DSRow, OutputMethod, Uint


DS_NAME: Final = "Suzhen/CodeChat-V2.0"
DS_REVISION: Final = "09dacf311596f8214075878600dcb60e5bcd7eb4"  # 2025-09-20
TARGET: Final = "train"


def _show_eg_convo(ds: Dataset, *, idx: Uint, output_method: OutputMethod = "both") -> None:
    """Show example conversation from original dataset."""
    assert idx < len(ds)

    convo = cast("DSRow", ds[idx])
    cout(Rule(f"[dim]Sample Conversation (model {convo['model']!r})[/]", style="dim"))

    if output_method in {"raw", "both"}:
        cout()
        cout(convo)

    if output_method == "raw":
        cout()
        return

    for turn in convo["conversation"]:
        for msg in turn:
            role, content = msg["role"], msg["content"]
            clr = "cyan" if role == "user" else "magenta"
            cout(f"\n[bold {clr}]● {role.capitalize()}[/]")
            cout(
                Padding(
                    Markdown(content),
                    pad=(0, 2),
                ),
            )
    cout()


def load_ds(
    *,
    overview: bool = False,
    eg_convo: int | None = None,
    eg_output_method: OutputMethod = "pretty",
) -> Dataset:
    """Load dataset.

    Keyword Args:
    (Optional)
        overview: Show dataset overview
        eg_convo: Index of example conversation to show
        eg_output_method: Output method for example ("raw", "pretty", or "both")
    """

    with cout.status("[bold green]Loading CodeChat-V2.0 dataset...", spinner="flip"):
        ds = load_dataset(DS_NAME, revision=DS_REVISION)["train"]

    if overview:
        show_ds_overview(ds, ds_name=DS_NAME)
    if eg_convo is not None:
        _show_eg_convo(ds, idx=eg_convo, output_method=eg_output_method)

    return ds
