from pathlib import Path
import pandas as pd

from models.vgg16_feature_maps import generate_vgg16_feature_maps
from models.vgg16_gradcam import generate_vgg16_gradcam


CORRECT_CSV = "outputs/vgg16_correct_subset.csv"
FAILURE_CSV = "outputs/vgg16_failure_subset.csv"

FEATUREMAP_OUTPUT_DIR = "outputs/vgg16_feature_maps"
GRADCAM_OUTPUT_DIR = "outputs/vgg16_gradcam"

OVERWRITE = False


def process_subset(csv_path: str, subset_name: str):
    csv_file = Path(csv_path)

    if not csv_file.exists():
        print(f"[SKIP] Missing CSV: {csv_path}")
        return

    df = pd.read_csv(csv_file)

    print(f"\n=== Processing {subset_name} subset ===")
    print(f"CSV: {csv_path}")
    print(f"Rows: {len(df)}")

    for i, row in df.iterrows():
        image_name = row["image_name"]
        thesis_class = row["thesis_class"]

        feature_dir = Path(FEATUREMAP_OUTPUT_DIR) / thesis_class / image_name.replace(".jpg", "")
        gradcam_dir = Path(GRADCAM_OUTPUT_DIR) / thesis_class / image_name.replace(".jpg", "")

        feature_done = (feature_dir / "late_feature_maps.png").exists()
        gradcam_done = (gradcam_dir / "gradcam_overlay.png").exists()

        if not OVERWRITE and feature_done and gradcam_done:
            print(f"[{i+1}/{len(df)}] SKIP {image_name} (already done)")
            continue

        print(f"[{i+1}/{len(df)}] Processing {image_name}")

        try:
            generate_vgg16_feature_maps(
                input_csv=csv_path,
                image_name=image_name,
                top_k=16,
                output_dir=FEATUREMAP_OUTPUT_DIR,
            )

            generate_vgg16_gradcam(
                input_csv=csv_path,
                image_name=image_name,
                output_dir=GRADCAM_OUTPUT_DIR,
            )

        except Exception as e:
            print(f"[ERROR] {image_name}: {e}")


def main():
    process_subset(CORRECT_CSV, "correct")
    process_subset(FAILURE_CSV, "failure")
    print("\nDone with VGG16 batch generation.")


if __name__ == "__main__":
    main()