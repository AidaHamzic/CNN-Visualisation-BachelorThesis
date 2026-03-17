def map_places365_to_thesis_class(label: str):
    label = label.lower().strip()

    CLASS_MAP = {
        "building": {
            "apartment-building-outdoor",
            "office-building",
            "skyscraper",
            "downtown",
            "building-facade",
            "hotel-outdoor",
            "house",
            "church",
            "tower",
            "skyscraper",

        },
        "forest": {
            "forest-broadleaf",
            "forest-path",
            "forest-road",
            "bamboo-forest",
            "rainforest",
            "woodland"
        },
        "glacier": {
            "glacier"
        },
        "mountain": {
            "mountain",
            "mountain-path",
            "mountain-snowy"
        },
        "water": {
            "ocean",
            "seashore",
            "coast",
            "lakeside",
            "river",
            "waterfall"
            "pond",
            "sea-cliff",

        },
        "street": {
            "street",
            "street-pedestrian",
            "street-urban",
            "highway",
            "road",
            "alley",


        }
    }

    for thesis_class, raw_labels in CLASS_MAP.items():
        if label in raw_labels:
            return thesis_class

    return None