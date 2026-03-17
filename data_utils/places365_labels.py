from data_utils.config import VAL_LABEL_FILE


def load_places365_labels():
    mapping = {}

    with open(VAL_LABEL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("/")
            class_name = parts[1]
            image_name = parts[2]

            mapping[image_name] = class_name

    return mapping