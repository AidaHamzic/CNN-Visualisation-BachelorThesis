from pathlib import Path
from data_utils.places365_labels import load_places365_labels
from data_utils.config import DATASET_ROOT


def main():

    labels = load_places365_labels()

    image_paths = sorted(DATASET_ROOT.glob("*.jpg"))

    print("images found:", len(image_paths))
    print("labels loaded:", len(labels))

    image_names = set(p.name for p in image_paths)
    label_names = set(labels.keys())

    missing_labels = image_names - label_names
    missing_images = label_names - image_names

    print()

    print("images without labels:", len(missing_labels))
    print("labels without images:", len(missing_images))

    if len(missing_labels) > 0:
        print("example missing label:", list(missing_labels)[:5])

    if len(missing_images) > 0:
        print("example missing image:", list(missing_images)[:5])

    print()

    if len(missing_labels) == 0 and len(missing_images) == 0:
        print("DATASET CONSISTENT ✓")
    else:
        print("DATASET INCONSISTENT")


if __name__ == "__main__":
    main()