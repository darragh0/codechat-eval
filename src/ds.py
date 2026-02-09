"""Dataset loading."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from datasets import List, Value, load_dataset

from utils.console import cout
from utils.display import maxlen

if TYPE_CHECKING:
    from datasets import Dataset


DS_NAME: Final = "Suzhen/CodeChat-V2.0"
DS_REVISION: Final = "09dacf311596f8214075878600dcb60e5bcd7eb4"  # 2025-09-20
TARGET: Final = "train"


def _show_ds_overview(ds: Dataset, ds_name: str) -> None:
    """Show original dataset overview."""

    def fmt_type(feat: List | Value | dict) -> str:
        if isinstance(feat, Value):
            return feat.dtype
        if isinstance(feat, List):
            return rf"list\[{fmt_type(feat.feature)}]"
        return type(feat).__name__

    def inner_fields(feat: List | Value | dict) -> dict | None:
        if isinstance(feat, List):
            return inner_fields(feat.feature)
        if isinstance(feat, dict):
            return feat
        return None

    cout(f"[dim]Dataset:[/] [bold]{ds_name!r}[/]")
    cout(f"[dim]Rows:[/] {len(ds):,}")
    cout("\n[dim]Features:[/]")

    w = maxlen(ds.features)
    zpad = len(str(len(ds.features)))

    for i, (name, feat) in enumerate(ds.features.items()):
        typ = fmt_type(feat)
        cout(f"  {i:0{zpad}}  {name:<{w}}\t[cyan]{typ}[/]")

        if inner := inner_fields(feat):
            items = list(inner.items())
            iw = max({w, maxlen(inner)}) - 4
            for j, (sub_name, sub_feat) in enumerate(items):
                pre = "└─" if j == len(items) - 1 else "├─"
                styp = fmt_type(sub_feat)
                cout(f"    {' ' * zpad}[dim]  {pre}[/] {sub_name:<{iw}}\t  [cyan]{styp}[/]")


def load_ds(*, overview: bool = False) -> Dataset:
    """Load dataset.

    Keyword Args:
    (Optional)
        overview: Show dataset overview
    """

    with cout.status("[bold green]Loading CodeChat-V2.0 dataset...", spinner="flip"):
        ds = load_dataset(DS_NAME, revision=DS_REVISION)["train"]

    if overview:
        _show_ds_overview(ds, ds_name=DS_NAME)

    return ds
