from collections.abc import Iterable, Sized


def maxlen(items: Iterable[Sized], /) -> int:
    """Return length of longest item in an iterable of sized objects."""
    return max(len(x) for x in items)
