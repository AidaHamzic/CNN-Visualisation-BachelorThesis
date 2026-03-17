from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data_utils.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from models.vgg16_model import load_pretrained_vgg16


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image_from_subset(csv_path: str, row_index: int = None, image_name: str = None):
    df = pd.read_csv(csv_path)

    if image_name is not None:
        matches = df[df["image_name"] == image_name]

        if matches.empty:
            raise ValueError(f"image_name '{image_name}' not found in {csv_path}")

        row = matches.iloc[0]

    elif row_index is not None:
        if row_index < 0 or row_index >= len(df):
            raise IndexError(
                f"ROW_INDEX {row_index} is out of range for {csv_path}. Total rows: {len(df)}"
            )
        row = df.iloc[row_index]

    else:
        raise ValueError("Provide either row_index or image_name.")

    image_path = row["image_path"]
    image_name = row["image_name"]
    thesis_class = row["thesis_class"]
    raw_places_label = row["true_scene_label"]
    aligned = row["semantically_aligned"]

    pil_image = Image.open(image_path).convert("RGB")
    resized_for_display = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))

    transform = get_transform()
    input_tensor = transform(pil_image).unsqueeze(0)

    rgb_float = np.array(resized_for_display).astype(np.float32) / 255.0

    return {
        "row": row,
        "image_path": image_path,
        "image_name": image_name,
        "thesis_class": thesis_class,
        "raw_places_label": raw_places_label,
        "aligned": aligned,
        "pil_image": pil_image,
        "display_image": resized_for_display,
        "rgb_float": rgb_float,
        "input_tensor": input_tensor,
    }


def get_target_layer(model):
    return model.features[28]


def generate_vgg16_gradcam(
    input_csv: str,
    row_index: int = None,
    image_name: str = None,
    output_dir: str = "outputs/vgg16_gradcam",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    sample = load_image_from_subset(
        input_csv,
        row_index=row_index,
        image_name=image_name
    )

    model, imagenet_classes = load_pretrained_vgg16()
    model = model.to(device)
    model.eval()

    input_tensor = sample["input_tensor"].to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities, k=5, dim=1)

    top1_idx = int(top5_indices[0][0].item())
    top1_label = imagenet_classes[top1_idx]
    top1_conf = round(float(top5_probs[0][0].item()), 4)

    top5_labels = [imagenet_classes[idx] for idx in top5_indices[0].cpu().tolist()]
    top5_scores = [round(float(x), 4) for x in top5_probs[0].cpu().tolist()]

    print("\nSelected sample")
    print("image_name:", sample["image_name"])
    print("image_path:", sample["image_path"])
    print("thesis_class:", sample["thesis_class"])
    print("raw_places_label:", sample["raw_places_label"])
    print("aligned:", sample["aligned"])
    print("top1 prediction:", top1_label, f"({top1_conf})")

    target_layer = get_target_layer(model)
    targets = [ClassifierOutputTarget(top1_idx)]

    sample_output_dir = (
        Path(output_dir)
        / sample["thesis_class"]
        / sample["image_name"].replace(".jpg", "")
    )
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]

    visualization = show_cam_on_image(
        sample["rgb_float"],
        grayscale_cam,
        use_rgb=True
    )

    original_path = sample_output_dir / "original_resized.png"
    overlay_path = sample_output_dir / "gradcam_overlay.png"
    heatmap_path = sample_output_dir / "gradcam_heatmap.png"

    sample["display_image"].save(original_path)

    overlay_image = Image.fromarray(visualization)
    overlay_image.save(overlay_path)

    heatmap_uint8 = (grayscale_cam * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(heatmap_uint8).save(heatmap_path)

    print("\nSaved outputs to:", sample_output_dir)

    return {
        "image_name": sample["image_name"],
        "image_path": sample["image_path"],
        "thesis_class": sample["thesis_class"],
        "raw_places_label": sample["raw_places_label"],
        "aligned": sample["aligned"],
        "top1_label": top1_label,
        "top1_confidence": top1_conf,
        "top5_labels": top5_labels,
        "top5_scores": top5_scores,
        "saved_files": {
            "original_image": str(original_path),
            "gradcam_overlay": str(overlay_path),
            "gradcam_heatmap": str(heatmap_path),
        },
    }


if __name__ == "__main__":
    result = generate_vgg16_gradcam(
        input_csv="outputs/vgg16_correct_subset.csv",
        row_index=0,
        image_name=None,
        output_dir="outputs/vgg16_gradcam",
    )

    print("\nDone.")
    print(result["saved_files"])