from pathlib import Path
from PIL import Image

from data_utils.config import DATASET_ROOT


def main():
    image_paths = sorted(DATASET_ROOT.glob("*.jpg"))

    print("dataset root:", DATASET_ROOT)
    print("jpg images found:", len(image_paths))

    if len(image_paths) == 0:
        print("NO IMAGES FOUND")
        return

    first_image = image_paths[0]
    print("first image:", first_image.name)

    try:
        with Image.open(first_image) as img:
            print("first image size:", img.size)
            print("first image mode:", img.mode)
    except Exception as e:
        print("failed to open first image:", e)


if __name__ == "__main__":
    main()