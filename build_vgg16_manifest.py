import csv
import json

import torch

from data_utils.config import OUTPUT_DIR
from data_utils.dataloaders import get_places365_dataloader
from data_utils.places365_labels import load_places365_labels
from data_utils.semantic_mapping import (
    SEMANTIC_MAP,
    is_semantically_aligned,
    normalize_label,
)
from models.vgg16_model import load_pretrained_vgg16

OUTPUT_MANIFEST = OUTPUT_DIR / "vgg16_places365_manifest.csv"
BATCH_SIZE = 8


def map_true_scene_to_thesis_class(true_scene_label: str):
    normalized_scene = normalize_label(true_scene_label)

    for thesis_class, allowed_labels in SEMANTIC_MAP.items():
        if normalized_scene in allowed_labels:
            return thesis_class

    return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    labels_map = load_places365_labels()
    dataloader = get_places365_dataloader(batch_size=BATCH_SIZE)

    model, imagenet_classes = load_pretrained_vgg16()
    model = model.to(device)
    model.eval()

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    scanned_images = 0
    skipped_missing_label = 0
    skipped_missing_thesis_class = 0

    with torch.no_grad():
        for images, image_names, image_paths in dataloader:
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            top5_confidences, top5_indices = torch.topk(probabilities, k=5, dim=1)

            for i in range(len(image_names)):
                scanned_images += 1

                image_name = image_names[i]
                image_path = image_paths[i]

                if image_name not in labels_map:
                    skipped_missing_label += 1
                    continue

                true_scene_label = labels_map[image_name]
                thesis_class = map_true_scene_to_thesis_class(true_scene_label)

                if thesis_class is None:
                    skipped_missing_thesis_class += 1
                    continue

                pred_idx = top5_indices[i][0].item()
                predicted_label = imagenet_classes[pred_idx]
                confidence = float(top5_confidences[i][0].item())

                top5_predictions = []
                for j in range(5):
                    cls_idx = top5_indices[i][j].item()
                    cls_name = imagenet_classes[cls_idx]
                    cls_conf = float(top5_confidences[i][j].item())

                    top5_predictions.append({
                        "rank": j + 1,
                        "label": cls_name,
                        "confidence": round(cls_conf, 4),
                    })

                semantically_aligned = is_semantically_aligned(
                    thesis_class=thesis_class,
                    top5_predictions=top5_predictions,
                )

                rows.append({
                    "image_name": image_name,
                    "image_path": image_path,
                    "true_scene_label": true_scene_label,
                    "thesis_class": thesis_class,
                    "predicted_label": predicted_label,
                    "confidence": round(confidence, 4),
                    "top5_predictions": json.dumps(top5_predictions, ensure_ascii=False),
                    "semantically_aligned": semantically_aligned,
                })

                if scanned_images % 100 == 0:
                    print(f"processed {scanned_images} images")

    with open(OUTPUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "image_path",
                "true_scene_label",
                "thesis_class",
                "predicted_label",
                "confidence",
                "top5_predictions",
                "semantically_aligned",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nsaved manifest to:", OUTPUT_MANIFEST)
    print("total images scanned:", scanned_images)
    print("total rows written:", len(rows))
    print("skipped missing image_name label:", skipped_missing_label)
    print("skipped missing thesis_class mapping:", skipped_missing_thesis_class)


if __name__ == "__main__":
    main()