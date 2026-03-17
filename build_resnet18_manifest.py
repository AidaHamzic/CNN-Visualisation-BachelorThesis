import csv
import json

import torch

from data_utils.config import OUTPUT_DIR
from data_utils.dataloaders import get_places365_dataloader
from data_utils.places365_labels import load_places365_labels
from data_utils.semantic_mapping import is_semantically_aligned
from data_utils.places_mapping import map_places365_to_thesis_class
from models.resnet18_model import load_pretrained_resnet18


OUTPUT_MANIFEST = OUTPUT_DIR / "resnet18_places365_manifest.csv"
BATCH_SIZE = 8


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    labels_map = load_places365_labels()
    dataloader = get_places365_dataloader(batch_size=BATCH_SIZE)

    model, imagenet_classes = load_pretrained_resnet18()
    model = model.to(device)

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    scanned_images = 0
    kept_images = 0

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
                    continue

                true_scene_label = labels_map[image_name]
                thesis_class = map_places365_to_thesis_class(true_scene_label)

                if thesis_class is None:
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
                        "confidence": round(cls_conf, 4)
                    })

                semantically_aligned = is_semantically_aligned(
                    thesis_class=thesis_class,
                    top5_predictions=top5_predictions
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

                kept_images += 1

                if scanned_images % 100 == 0:
                    print(f"processed {scanned_images} images | kept {kept_images}")

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
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nsaved manifest to:", OUTPUT_MANIFEST)
    print("total images processed:", scanned_images)
    print("total rows written:", len(rows))


if __name__ == "__main__":
    main()