import pandas as pd
from data_utils.places_mapping import map_places365_to_thesis_class


INPUT_PATH = "outputs/vgg16_places365_manifest.csv"
OUTPUT_PATH = "outputs/vgg16_places365_manifest_FIXED.csv"


def main():
    print("Loading manifest...")
    df = pd.read_csv(INPUT_PATH)

    print("Total rows:", len(df))

    if "true_scene_label" not in df.columns:
        raise ValueError("ERROR: true_scene_label column is missing")

    print("Mapping thesis_class...")

    df["thesis_class"] = df["true_scene_label"].apply(
        map_places365_to_thesis_class
    )

    missing = df["thesis_class"].isna().sum()

    print("Missing thesis_class:", missing)

    print("Saving fixed manifest...")
    df.to_csv(OUTPUT_PATH, index=False)

    print("Done.")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()