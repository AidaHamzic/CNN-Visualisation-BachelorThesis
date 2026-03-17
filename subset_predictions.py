import json
import pandas as pd

INPUT_CSV = "outputs/vgg16_places365_subset_manifest.csv"

THESIS_CLASS = "mountain"
N = 20


def main():
    df = pd.read_csv(INPUT_CSV)

    subset = df[df["thesis_class"] == THESIS_CLASS].head(N)

    print(f"\nThesis class: {THESIS_CLASS}")
    print(f"Rows shown: {len(subset)}\n")

    for _, row in subset.iterrows():
        top5 = json.loads(row["top5_predictions"])
        labels = [item["label"] for item in top5]

        print(f"image: {row['image_name']}")
        print(f"raw Places365 label: {row['true_scene_label']}")
        print(f"top5: {labels}")
        print(f"aligned: {row['semantically_aligned']}")
        print("-" * 60)


if __name__ == "__main__":
    main()