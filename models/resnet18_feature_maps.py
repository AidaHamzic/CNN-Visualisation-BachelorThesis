from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from data_utils.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from models.resnet18_model import load_pretrained_resnet18


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

    return {
        "row": row,
        "image_path": image_path,
        "image_name": image_name,
        "thesis_class": thesis_class,
        "raw_places_label": raw_places_label,
        "aligned": aligned,
        "pil_image": pil_image,
        "display_image": resized_for_display,
        "input_tensor": input_tensor,
    }


def get_resnet18_target_layers(model):
    """
    ResNet-18 structure for torchvision==0.17.1
    Representative stages:
    early  = layer1
    middle = layer3
    late   = layer4
    """
    return {
        "early": model.layer1,
        "middle": model.layer3,
        "late": model.layer4,
    }


def register_hooks(target_layers, activations_dict):
    hooks = []

    def make_hook(layer_name):
        def hook_fn(module, input_tensor, output_tensor):
            activations_dict[layer_name] = output_tensor.detach().cpu()
        return hook_fn

    for layer_name, layer in target_layers.items():
        hooks.append(layer.register_forward_hook(make_hook(layer_name)))

    return hooks


def select_top_k_channels(feature_tensor: torch.Tensor, top_k: int = 16):
    feature_tensor = feature_tensor.squeeze(0)  # [C, H, W]
    channel_means = feature_tensor.mean(dim=(1, 2))
    top_k = min(top_k, feature_tensor.shape[0])

    topk_indices = torch.topk(channel_means, k=top_k).indices.tolist()
    selected_maps = feature_tensor[topk_indices]

    return selected_maps, topk_indices


def normalize_feature_map(feature_map: np.ndarray):
    min_val = feature_map.min()
    max_val = feature_map.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(feature_map)

    return (feature_map - min_val) / (max_val - min_val)


def save_feature_grid(selected_maps: torch.Tensor, layer_name: str, save_path: Path):
    selected_maps = selected_maps.numpy()

    num_maps = selected_maps.shape[0]
    cols = 4
    rows = int(np.ceil(num_maps / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")

        if idx < num_maps:
            fm = normalize_feature_map(selected_maps[idx])
            ax.imshow(fm, cmap="viridis")
            ax.set_title(f"ch {idx+1}", fontsize=8)

    fig.suptitle(f"ResNet-18 {layer_name} feature maps (top-{num_maps})", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_resnet18_feature_maps(
    input_csv: str,
    row_index: int = None,
    image_name: str = None,
    top_k: int = 16,
    output_dir: str = "outputs/resnet18_feature_maps",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    sample = load_image_from_subset(
        input_csv,
        row_index=row_index,
        image_name=image_name
    )

    model, imagenet_classes = load_pretrained_resnet18()
    model = model.to(device)
    model.eval()

    target_layers = get_resnet18_target_layers(model)

    activations = {}
    hooks = register_hooks(target_layers, activations)

    input_tensor = sample["input_tensor"].to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top5_probs, top5_indices = torch.topk(probabilities, k=5, dim=1)

    for h in hooks:
        h.remove()

    top5_labels = [imagenet_classes[idx] for idx in top5_indices[0].cpu().tolist()]
    top5_scores = [round(float(x), 4) for x in top5_probs[0].cpu().tolist()]

    print("\nSelected sample")
    print("image_name:", sample["image_name"])
    print("image_path:", sample["image_path"])
    print("thesis_class:", sample["thesis_class"])
    print("raw_places_label:", sample["raw_places_label"])
    print("aligned:", sample["aligned"])
    print("\nTop-5 predictions:")
    for rank, (label, score) in enumerate(zip(top5_labels, top5_scores), start=1):
        print(f"{rank}. {label} ({score})")

    sample_output_dir = (
        Path(output_dir)
        / sample["thesis_class"]
        / sample["image_name"].replace(".jpg", "")
    )
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    sample["display_image"].save(sample_output_dir / "original_resized.png")

    saved_files = {
        "original_image": str(sample_output_dir / "original_resized.png"),
        "feature_maps": {}
    }

    for layer_name, feature_tensor in activations.items():
        selected_maps, selected_indices = select_top_k_channels(feature_tensor, top_k=top_k)

        print(f"\nLayer: {layer_name}")
        print("selected channel indices:", selected_indices)

        save_path = sample_output_dir / f"{layer_name}_feature_maps.png"
        save_feature_grid(selected_maps, layer_name, save_path)

        saved_files["feature_maps"][layer_name] = {
            "path": str(save_path),
            "selected_channel_indices": selected_indices,
        }

    print("\nSaved outputs to:", sample_output_dir)

    return {
        "image_name": sample["image_name"],
        "image_path": sample["image_path"],
        "thesis_class": sample["thesis_class"],
        "raw_places_label": sample["raw_places_label"],
        "aligned": sample["aligned"],
        "top5_labels": top5_labels,
        "top5_scores": top5_scores,
        "saved_files": saved_files,
    }


if __name__ == "__main__":
    result = generate_resnet18_feature_maps(
        input_csv="outputs/resnet18_correct_subset.csv",
        row_index=0,
        image_name=None,
        top_k=16,
        output_dir="outputs/resnet18_feature_maps",
    )

    print("\nDone.")
    print(result["saved_files"])