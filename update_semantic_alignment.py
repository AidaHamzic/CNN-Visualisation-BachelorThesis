#import pandas as pd
#df = pd.read_csv("outputs/vgg16_places365_manifest.csv")
#print(df.columns)


import json
import pandas as pd
from data_utils.semantic_mapping import is_semantically_aligned


def safe_parse_top5(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        return json.loads(value)
    except Exception:
        return []


def update_alignment(csv_path):
    df = pd.read_csv(csv_path)

    if "thesis_class" not in df.columns:
        raise ValueError(f"'thesis_class' column not found in {csv_path}")

    if "top5_predictions" in df.columns:
        df["top5_predictions"] = df["top5_predictions"].apply(safe_parse_top5)
    else:
        df["top5_predictions"] = [[] for _ in range(len(df))]

    df["semantically_aligned"] = df.apply(
        lambda row: is_semantically_aligned(
            row["thesis_class"],
            row["top5_predictions"]
        ),
        axis=1
    )

    df.to_csv(csv_path, index=False)
    print(f"Updated: {csv_path}")


update_alignment("outputs/vgg16_correct_subset.csv")
update_alignment("outputs/vgg16_failure_subset.csv")
update_alignment("outputs/resnet18_correct_subset.csv")
update_alignment("outputs/resnet18_failure_subset.csv")