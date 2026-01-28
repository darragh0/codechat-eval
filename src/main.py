from ds import load_ds
from filter import filter_ds


def main() -> None:
    ds = load_ds(overview=True, eg_convo=0, eg_output_method="raw")
    filtered_ds = filter_ds(ds, overview=True, only_english=True, langs=("python", "py"))

    # for row in rows_as(filtered_ds, FilteredDSRow):
    #     print(row["prompt"])


if __name__ == "__main__":
    main()
