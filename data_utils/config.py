from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASET_ROOT = PROJECT_ROOT / "val_256"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VAL_LABEL_FILE = PROJECT_ROOT / "data" / "val.txt"

