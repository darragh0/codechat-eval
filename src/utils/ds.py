"""Dataset utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from datasets import Dataset, List, Value

from utils.console import cout
from utils.misc import maxlen

if TYPE_CHECKING:
    from collections.abc import Iterator


def show_ds_overview(ds: Dataset, ds_name: str) -> None:
    """Show dataset overview.

    Args:
        ds: Dataset instance
        ds_name: Dataset name
    """

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
    cout()


def rows_as[T](ds: Dataset, _: type[T], /) -> Iterator[T]:
    """Cast dataset rows to type `T`."""
    for r in ds:
        yield cast("T", r)
