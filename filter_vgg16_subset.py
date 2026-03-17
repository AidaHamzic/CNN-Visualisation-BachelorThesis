import json
import pandas as pd

from data_utils.places_mapping import map_places365_to_thesis_class
from data_utils.semantic_mapping import is_semantically_aligned


INPUT_CSV = "outputs/vgg16_places365_manifest.csv"
OUTPUT_CSV = "outputs/vgg16_places365_subset_manifest.csv"
CORRECT_CSV = "outputs/vgg16_correct_subset.csv"
FAILURE_CSV = "outputs/vgg16_failure_subset.csv"

PER_CLASS_LIMIT = 20


def main():
    df = pd.read_csv(INPUT_CSV)

    print("original rows:", len(df))

    df["thesis_class"] = df["true_scene_label"].apply(map_places365_to_thesis_class)

    df = df[df["thesis_class"].notna()].copy()

    print("rows after thesis-class filtering:", len(df))
    print("\nclass counts:")
    print(df["thesis_class"].value_counts())

    df["semantically_aligned"] = df.apply(
        lambda row: is_semantically_aligned(
            thesis_class=row["thesis_class"],
            top5_predictions=json.loads(row["top5_predictions"])
        ),
        axis=1
    )

    df.to_csv(OUTPUT_CSV, index=False)

    print("\nsubset manifest saved to:", OUTPUT_CSV)

    print("\nalignment rate by thesis class:")
    print(df.groupby("thesis_class")["semantically_aligned"].mean())

    correct_df = df[df["semantically_aligned"] == True].copy()
    failure_df = df[df["semantically_aligned"] == False].copy()

    correct_subset = (
        correct_df.groupby("thesis_class", group_keys=False)
        .head(PER_CLASS_LIMIT)
        .copy()
    )

    failure_subset = (
        failure_df.groupby("thesis_class", group_keys=False)
        .head(PER_CLASS_LIMIT)
        .copy()
    )

    correct_subset.to_csv(CORRECT_CSV, index=False)
    failure_subset.to_csv(FAILURE_CSV, index=False)

    print("\ncorrect subset saved to:", CORRECT_CSV)
    print("failure subset saved to:", FAILURE_CSV)

    print("\ncorrect subset counts:")
    print(correct_subset["thesis_class"].value_counts())

    print("\nfailure subset counts:")
    print(failure_subset["thesis_class"].value_counts())


if __name__ == "__main__":
    main()