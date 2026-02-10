from ds import load_ds
from filter import filter_ds
from semantic import analyse_semantics
from syntax import analyse_syntax
from utils.console import cout


def main() -> None:
    cout.clear()
    ds = load_ds(overview=True)
    df = filter_ds(ds, overview=True)
    df = analyse_syntax(df, overview=True)
    df = analyse_semantics(df, overview=True)


if __name__ == "__main__":
    main()
