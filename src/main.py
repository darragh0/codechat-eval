from ds import load_ds
from filter import filter_ds
from syntax import analyse_syntax

# from utils.console import cout


def main() -> None:
    ds = load_ds(overview=True)
    cache_key, df = filter_ds(ds, overview=True, only_english=True, langs=("python", "py"))
    df = analyse_syntax(df, overview=True, cache_key=cache_key)
    # cout(df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
