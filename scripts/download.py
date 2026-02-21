#!/usr/bin/env python3

"""Load the dataset from huggingface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from datasets import List, Value, load_dataset
from utils.cache import CACHE_DIR
from utils.console import cout
from utils.display import maxlen, prettypath

if TYPE_CHECKING:
    from datasets import Dataset


DS_NAME: Final = "Suzhen/CodeChat-V2.0"
DS_REVISION: Final = "09dacf311596f8214075878600dcb60e5bcd7eb4"  # 2025-09-20
TARGET: Final = "train"


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


def show_oview(ds: Dataset, ds_name: str) -> None:
    """Show original dataset overview."""

    cout(f"[dim]Dataset[/]    {ds_name!r}")
    location = prettypath(CACHE_DIR / f"{ds_name}.parquet")
    cout(f"[dim]Location[/]   {location}")
    cout(f"[dim]Rows[/]       {len(ds):,}")
    cout("\n[dim]Features[/]")

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
    cout()


def load_ds(*, overview: bool = True) -> Dataset:
    with cout.status("[dim]Loading CodeChat-V2.0 dataset[/]", spinner="flip"):
        ds = load_dataset(DS_NAME, revision=DS_REVISION, cache_dir=str(CACHE_DIR))["train"]

    if overview:
        show_oview(ds, ds_name=DS_NAME)
    return ds


def main() -> None:
    load_ds()


if __name__ == "__main__":
    from utils.cache import graceful_exit

    with graceful_exit("download stopped"):
        main()
