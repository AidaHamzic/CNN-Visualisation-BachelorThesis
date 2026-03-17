import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from PIL import Image

from models.model_registry import MODEL_CONFIG


st.set_page_config(
    page_title="CNN Visual Analysis Interface",
    layout="wide",
    initial_sidebar_state="expanded",
)


def safe_parse_top5(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        return json.loads(value)
    except Exception:
        return []


def normalize_bool_col(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if col_name in df.columns:
        df[col_name] = df[col_name].astype(str).str.lower().eq("true")
    return df


def format_confidence(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "n/a"


@st.cache_data
def load_csv(csv_path_str: str):
    csv_path = Path(csv_path_str)

    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    if "top5_predictions" in df.columns:
        df["top5_predictions"] = df["top5_predictions"].apply(safe_parse_top5)
    else:
        df["top5_predictions"] = [[] for _ in range(len(df))]

    df = normalize_bool_col(df, "semantically_aligned")
    return df


def get_csv_path(model_key: str, subset_mode: str):
    model_cfg = MODEL_CONFIG[model_key]

    if subset_mode == "correct":
        return PROJECT_ROOT / model_cfg["correct_csv"]
    if subset_mode == "failure":
        return PROJECT_ROOT / model_cfg["failure_csv"]
    return PROJECT_ROOT / model_cfg["subset_csv"]


def get_manifest_path(model_key: str):
    model_cfg = MODEL_CONFIG[model_key]

    if "manifest_csv" in model_cfg:
        return PROJECT_ROOT / model_cfg["manifest_csv"]

    fallback = PROJECT_ROOT / "outputs" / f"{model_key}_places365_manifest.csv"
    return fallback


def get_filtered_df(model_key: str, subset_mode: str, thesis_class: str):
    csv_path = get_csv_path(model_key, subset_mode)
    df = load_csv(str(csv_path))

    if df.empty:
        return df

    if thesis_class != "all" and "thesis_class" in df.columns:
        df = df[df["thesis_class"] == thesis_class].copy()

    return df.reset_index(drop=True)


@st.cache_data
def get_manifest_df(model_key: str):
    manifest_path = get_manifest_path(model_key)
    df = load_csv(str(manifest_path))
    return df


def build_saved_outputs(model_key: str, thesis_class: str, image_name: str):
    image_stem = Path(image_name).stem

    feature_base = PROJECT_ROOT / "outputs" / f"{model_key}_feature_maps" / thesis_class / image_stem
    gradcam_base = PROJECT_ROOT / "outputs" / f"{model_key}_gradcam" / thesis_class / image_stem

    feature_saved_files = {
        "original_image": str(feature_base / "original_resized.png"),
        "feature_maps": {
            "early": {"path": str(feature_base / "early_feature_maps.png")},
            "middle": {"path": str(feature_base / "middle_feature_maps.png")},
            "late": {"path": str(feature_base / "late_feature_maps.png")},
        },
    }

    gradcam_saved_files = {
        "original_image": str(gradcam_base / "original_resized.png"),
        "gradcam_overlay": str(gradcam_base / "gradcam_overlay.png"),
        "gradcam_heatmap": str(gradcam_base / "gradcam_heatmap.png"),
    }

    return feature_saved_files, gradcam_saved_files


def render_top5(top5_preds):
    if not top5_preds:
        st.warning("No top-5 predictions found for this row.")
        return

    for rank, pred in enumerate(top5_preds, start=1):
        label = pred.get("label", "unknown")
        conf = pred.get("confidence", None)

        if conf is None:
            st.write(f"{rank}. **{label}**")
        else:
            st.write(f"{rank}. **{label}** ({format_confidence(conf)})")


def render_feature_maps(saved_files: dict):
    feature_maps = saved_files.get("feature_maps", {})
    ordered_layers = ["early", "middle", "late"]
    cols = st.columns(3)

    for col, layer_name in zip(cols, ordered_layers):
        with col:
            layer_info = feature_maps.get(layer_name, {})
            img_path = layer_info.get("path")

            st.markdown(f"**{layer_name.capitalize()} layer**")

            if img_path and Path(img_path).exists():
                st.image(img_path, caption=f"{layer_name} feature maps", use_container_width=True)
            else:
                st.info(f"No saved {layer_name} feature maps.")


def render_gradcam(saved_files: dict):
    cols = st.columns(3)

    with cols[0]:
        original_path = saved_files.get("original_image")
        if original_path and Path(original_path).exists():
            st.image(original_path, caption="Original resized image", use_container_width=True)
        else:
            st.info("No saved original image.")

    with cols[1]:
        overlay_path = saved_files.get("gradcam_overlay")
        if overlay_path and Path(overlay_path).exists():
            st.image(overlay_path, caption="Grad-CAM overlay", use_container_width=True)
        else:
            st.info("No saved Grad-CAM overlay.")

    with cols[2]:
        heatmap_path = saved_files.get("gradcam_heatmap")
        if heatmap_path and Path(heatmap_path).exists():
            st.image(heatmap_path, caption="Grad-CAM heatmap", use_container_width=True)
        else:
            st.info("No saved Grad-CAM heatmap.")


def render_metadata_block_single(row):
    st.write(f"**Raw Places365 label:** {row.get('true_scene_label', 'n/a')}")
    st.write(f"**Thesis class:** {row.get('thesis_class', 'n/a')}")
    st.write(f"**Semantically aligned:** {bool(row.get('semantically_aligned', False))}")
    st.write(f"**Top-1 prediction:** {row.get('predicted_label', 'n/a')}")
    st.write(f"**Confidence:** {format_confidence(row.get('confidence', None))}")


def render_metadata_block_compare(row, prefix: str):
    st.write(f"**Raw Places365 label:** {row.get(f'true_scene_label_{prefix}', 'n/a')}")
    st.write(f"**Thesis class:** {row.get(f'thesis_class_{prefix}', 'n/a')}")
    st.write(f"**Aligned:** {bool(row.get(f'semantically_aligned_{prefix}', False))}")
    st.write(f"**Top-1 prediction:** {row.get(f'predicted_label_{prefix}', 'n/a')}")
    st.write(f"**Confidence:** {format_confidence(row.get(f'confidence_{prefix}', None))}")


def render_compact_summary(row, prefix: str):
    pred = row.get(f"predicted_label_{prefix}", "unknown")
    conf = format_confidence(row.get(f"confidence_{prefix}", None))
    aligned = bool(row.get(f"semantically_aligned_{prefix}", False))
    st.caption(f"Top-1: {pred} | Confidence: {conf} | Aligned: {aligned}")


@st.cache_data
def get_compare_merged_df():
    vgg_df = get_manifest_df("vgg16").copy()
    resnet_df = get_manifest_df("resnet18").copy()

    if vgg_df.empty or resnet_df.empty:
        return pd.DataFrame()

    needed_cols = [
        "image_name",
        "image_path",
        "true_scene_label",
        "thesis_class",
        "predicted_label",
        "confidence",
        "top5_predictions",
        "semantically_aligned",
    ]

    vgg_keep = [c for c in needed_cols if c in vgg_df.columns]
    resnet_keep = [c for c in needed_cols if c in resnet_df.columns]

    vgg_df = vgg_df[vgg_keep].copy()
    resnet_df = resnet_df[resnet_keep].copy()

    merged = vgg_df.merge(
        resnet_df,
        on="image_name",
        how="inner",
        suffixes=("_vgg", "_resnet"),
    )

    return merged


def get_compare_filtered_df(case_type: str, thesis_class: str):
    merged = get_compare_merged_df()

    if merged.empty:
        return merged

    if thesis_class != "all":
        if "thesis_class_vgg" in merged.columns:
            merged = merged[merged["thesis_class_vgg"] == thesis_class].copy()

    if case_type == "All shared images":
        return merged.reset_index(drop=True)

    if case_type == "VGG16 aligned, ResNet-18 not aligned":
        merged = merged[
            (merged["semantically_aligned_vgg"] == True)
            & (merged["semantically_aligned_resnet"] == False)
        ].copy()
        return merged.reset_index(drop=True)

    if case_type == "ResNet-18 aligned, VGG16 not aligned":
        merged = merged[
            (merged["semantically_aligned_vgg"] == False)
            & (merged["semantically_aligned_resnet"] == True)
        ].copy()
        return merged.reset_index(drop=True)

    if case_type == "Both aligned":
        merged = merged[
            (merged["semantically_aligned_vgg"] == True)
            & (merged["semantically_aligned_resnet"] == True)
        ].copy()
        return merged.reset_index(drop=True)

    if case_type == "Both not aligned":
        merged = merged[
            (merged["semantically_aligned_vgg"] == False)
            & (merged["semantically_aligned_resnet"] == False)
        ].copy()
        return merged.reset_index(drop=True)

    return merged.reset_index(drop=True)


def main():
    st.title("CNN Visual Analysis Interface")
    st.caption("Viewer for precomputed prediction, feature maps, Grad-CAM, and side-by-side comparison.")

    st.sidebar.header("Controls")

    mode = st.sidebar.radio(
        label="Mode",
        options=["Single Model", "Compare Models"],
        index=0,
    )

    subset_mode = st.sidebar.selectbox(
        label="Subset source",
        options=["correct", "failure", "all"],
        index=0,
    )

    if mode == "Single Model":
        selected_model_key = st.sidebar.selectbox(
            label="Model",
            options=list(MODEL_CONFIG.keys()),
            format_func=lambda x: MODEL_CONFIG[x]["display_name"],
            index=0,
        )

        df = get_filtered_df(selected_model_key, subset_mode, thesis_class="all")

        if df.empty:
            st.error("No data found for the selected model/subset.")
            st.stop()

        thesis_classes = sorted(df["thesis_class"].dropna().unique().tolist()) if "thesis_class" in df.columns else []

        selected_class = st.sidebar.selectbox(
            label="Thesis class",
            options=["all"] + thesis_classes,
            index=0,
        )

        filtered_df = get_filtered_df(selected_model_key, subset_mode, selected_class)

        if filtered_df.empty:
            st.warning("No rows match the current filters.")
            st.stop()

        image_names = filtered_df["image_name"].tolist()
        selected_image_name = st.sidebar.selectbox(
            label="Select image",
            options=image_names,
            index=0,
        )

        selected_row = filtered_df[filtered_df["image_name"] == selected_image_name].iloc[0]
        model_cfg = MODEL_CONFIG[selected_model_key]

        total_rows = len(filtered_df)
        aligned_count = int(filtered_df["semantically_aligned"].sum()) if "semantically_aligned" in filtered_df.columns else 0
        failure_count = total_rows - aligned_count

        m1, m2, m3 = st.columns(3)
        m1.metric("Filtered rows", total_rows)
        m2.metric("Aligned", aligned_count)
        m3.metric("Failure", failure_count)

        st.markdown("---")

        image_path = Path(selected_row["image_path"])
        if not image_path.exists():
            st.error(f"Image file does not exist: {image_path}")
            st.stop()

        pil_img = Image.open(image_path).convert("RGB")

        left, right = st.columns([1.15, 0.85])

        with left:
            st.subheader(f"{model_cfg['display_name']} | Selected image")
            st.image(pil_img, caption=selected_row["image_name"], use_container_width=True)

        with right:
            st.subheader("Summary")
            render_metadata_block_single(selected_row)

        st.markdown("---")
        st.subheader("Top-5 predictions")
        render_top5(selected_row["top5_predictions"])

        feature_saved_files, gradcam_saved_files = build_saved_outputs(
            selected_model_key,
            selected_row["thesis_class"],
            selected_row["image_name"],
        )

        tabs = st.tabs(["Feature Maps", "Grad-CAM", "Notes"])

        with tabs[0]:
            render_feature_maps(feature_saved_files)

        with tabs[1]:
            render_gradcam(gradcam_saved_files)
            st.write(f"**Top-1 label used for Grad-CAM:** {selected_row.get('predicted_label', 'n/a')}")
            st.write(f"**Top-1 confidence:** {format_confidence(selected_row.get('confidence', None))}")

        with tabs[2]:
            st.subheader("Method notes")
            st.write(
                "- Feature maps are loaded from saved outputs for early, middle, and late layers.\n"
                "- Grad-CAM is loaded from saved outputs generated during the batch step.\n"
                "- This interface is a viewer of precomputed results.\n"
                "- Single-model mode uses the curated subset CSVs."
            )

    else:
        merged_all = get_compare_merged_df()

        if merged_all.empty:
            st.error("Comparison data could not be loaded.")
            st.stop()

        thesis_classes = []
        if "thesis_class_vgg" in merged_all.columns:
            thesis_classes = sorted(merged_all["thesis_class_vgg"].dropna().unique().tolist())

        selected_class = st.sidebar.selectbox(
            label="Thesis class",
            options=["all"] + thesis_classes,
            index=0,
        )

        case_type = st.sidebar.selectbox(
            label="Case type",
            options=[
                "All shared images",
                "VGG16 aligned, ResNet-18 not aligned",
                "ResNet-18 aligned, VGG16 not aligned",
                "Both aligned",
                "Both not aligned",
            ],
            index=0,
        )

        compare_df = get_compare_filtered_df(case_type, selected_class)

        if compare_df.empty:
            st.warning("No images found for the selected comparison filters.")
            st.stop()

        image_names = compare_df["image_name"].tolist()
        selected_image_name = st.sidebar.selectbox(
            label="Select image",
            options=image_names,
            index=0,
        )

        row = compare_df[compare_df["image_name"] == selected_image_name].iloc[0]

        st.subheader("Model Comparison")
        st.caption(case_type)

        model_specs = [
            ("vgg16", "vgg"),
            ("resnet18", "resnet"),
        ]

        cols = st.columns(2)

        for col, (model_key, prefix) in zip(cols, model_specs):
            model_cfg = MODEL_CONFIG[model_key]

            thesis_class_value = row.get(f"thesis_class_{prefix}", "unknown")
            image_name_value = row["image_name"]

            feature_saved_files, gradcam_saved_files = build_saved_outputs(
                model_key,
                thesis_class_value,
                image_name_value,
            )

            with col:
                st.markdown(f"## {model_cfg['display_name']}")
                render_compact_summary(row, prefix)

                image_path_value = row.get(f"image_path_{prefix}", None)
                if image_path_value and Path(image_path_value).exists():
                    st.image(image_path_value, caption=image_name_value, use_container_width=True)
                else:
                    st.info("Original image not found.")

                render_metadata_block_compare(row, prefix)

                st.markdown("**Top-5 predictions**")
                render_top5(row.get(f"top5_predictions_{prefix}", []))

                inner_tabs = st.tabs(["Feature Maps", "Grad-CAM"])

                with inner_tabs[0]:
                    render_feature_maps(feature_saved_files)

                with inner_tabs[1]:
                    render_gradcam(gradcam_saved_files)
                    st.write(f"**Top-1 label used for Grad-CAM:** {row.get(f'predicted_label_{prefix}', 'n/a')}")
                    st.write(f"**Top-1 confidence:** {format_confidence(row.get(f'confidence_{prefix}', None))}")


if __name__ == "__main__":
    main()