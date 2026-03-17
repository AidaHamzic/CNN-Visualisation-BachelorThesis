import pandas as pd


def load_subset_pair(correct_csv: str, failure_csv: str) -> pd.DataFrame:
    correct_df = pd.read_csv(correct_csv)
    failure_df = pd.read_csv(failure_csv)

    df = pd.concat([correct_df, failure_df], ignore_index=True)

    df["semantically_aligned"] = (
        df["semantically_aligned"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    return df


def main():
    vgg = load_subset_pair(
        "outputs/vgg16_correct_subset.csv",
        "outputs/vgg16_failure_subset.csv"
    )

    resnet = load_subset_pair(
        "outputs/resnet18_correct_subset.csv",
        "outputs/resnet18_failure_subset.csv"
    )

    merged = vgg[[
        "image_name",
        "thesis_class",
        "true_scene_label",
        "predicted_label",
        "confidence",
        "semantically_aligned"
    ]].rename(columns={
        "predicted_label": "vgg_predicted_label",
        "confidence": "vgg_confidence",
        "semantically_aligned": "vgg_aligned"
    }).merge(
        resnet[[
            "image_name",
            "thesis_class",
            "true_scene_label",
            "predicted_label",
            "confidence",
            "semantically_aligned"
        ]].rename(columns={
            "predicted_label": "resnet_predicted_label",
            "confidence": "resnet_confidence",
            "semantically_aligned": "resnet_aligned"
        }),
        on=["image_name", "thesis_class", "true_scene_label"],
        how="inner"
    )

    vgg_semantic_alignment_only = merged[
        (merged["vgg_aligned"] == True) & (merged["resnet_aligned"] == False)
    ].copy()

    resnet_semantic_alignment_only = merged[
        (merged["vgg_aligned"] == False) & (merged["resnet_aligned"] == True)
    ].copy()

    vgg_semantic_alignment_only.to_csv(
        "outputs/vgg_semantic_alignment_only.csv",
        index=False
    )

    resnet_semantic_alignment_only.to_csv(
        "outputs/resnet_semantic_alignment_only.csv",
        index=False
    )

    print("\nVGG16 semantically aligned, ResNet-18 not aligned:")
    if vgg_semantic_alignment_only.empty:
        print("No cases found.")
    else:
        print(vgg_semantic_alignment_only[[
            "image_name",
            "thesis_class",
            "true_scene_label",
            "vgg_predicted_label",
            "vgg_confidence",
            "resnet_predicted_label",
            "resnet_confidence"
        ]].to_string(index=False))

    print("\nResNet-18 semantically aligned, VGG16 not aligned:")
    if resnet_semantic_alignment_only.empty:
        print("No cases found.")
    else:
        print(resnet_semantic_alignment_only[[
            "image_name",
            "thesis_class",
            "true_scene_label",
            "vgg_predicted_label",
            "vgg_confidence",
            "resnet_predicted_label",
            "resnet_confidence"
        ]].to_string(index=False))

    total = len(merged)

    print("\nCounts:")
    print(f"VGG16 semantic-alignment-only cases: {len(vgg_semantic_alignment_only)}")
    print(f"ResNet-18 semantic-alignment-only cases: {len(resnet_semantic_alignment_only)}")

    print("\nSummary:")
    print(f"Total compared images: {total}")
    if total > 0:
        print(
            f"VGG16 semantic-alignment-only proportion: "
            f"{len(vgg_semantic_alignment_only) / total:.2%}"
        )
        print(
            f"ResNet-18 semantic-alignment-only proportion: "
            f"{len(resnet_semantic_alignment_only) / total:.2%}"
        )
    else:
        print("No overlapping images found for comparison.")


if __name__ == "__main__":
    main()