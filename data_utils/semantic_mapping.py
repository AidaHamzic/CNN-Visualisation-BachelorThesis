def normalize_label(label: str) -> str:
    label = label.lower().strip()


    if "/" in label:
        label = label.split("/")[-1]


    label = label.replace(" ", "-")

    return label


SEMANTIC_MAP = {
    "building": {
        "building",
        "skyscraper",
        "palace",
        "castle",
        "monastery",
        "church",
        "mosque",
        "restaurant",
        "house",
        "hotel",
        "library",
        "planetarium",
        "barn",
"solar-dish",
"dome",
"greenhouse",
"prison",
"boathouse",
"patio",
"tile-roof",
"bell-cote",
"obelisk",
"triumphal-arch",
"cinema",
"library",
"bookshop",
"mobile-home",
"tobacco-shop",
"toyshop",
    },

    "forest": {
        "forest",
        "rainforest",
        "wood",
        "grove",
        "park",
"valley",
    "cliff",
    "mountain-tent",
    "wood-rabbit",
    "bison",
    "ox",
    "brown-bear",
    "wild-boar",
    "ibex",
    "spider-monkey",
    "capuchin",
    "squirrel-monkey",
    "three-toed-sloth",
    "gibbon",
    "howler-monkey",
    "orangutan",
    "langur",
    "macaw"

    },

    "glacier": {
        "alp",
        "iceberg",
        "snowfield",
        "mountain",
        "valley",
        "cliff",
        "ski",
        "snowplow",
        "dogsled"
    },

    "mountain": {
        "alp",
        "mountain",
        "valley",
        "cliff",
        "ridge",
        "volcano",
        "mountain-tent",
        "mountain-bike",

    },

    "water": {
        "seashore",
        "lakeside",
        "coast",
        "shore",
        "sandbar",
        "breakwater",
        "geyser",
        "dock",
        "wreck",
        "trimaran",
        "catamaran",
        "speedboat",
        "yawl",
        "lifeboat",
        "paddle",
        "canoe",
        "gondola",
        "beacon",
        "boathouse",
        "snorkel",
        "coral-reef",
        "goldfish",
        "oystercatcher",
        "killer-whale",
    },

    "street": {
        "street-sign",
        "traffic-light",
        "cab",
        "parking-meter",
        "crosswalk",
        "streetcar",
        "trolleybus",
        "police-van",
        "passenger-car",
        "limousine",
        "jinrikisha",
        "tow-truck",
        "moving-van",
        "trailer-truck",
        "garbage-truck",

    }
}


def is_semantically_aligned(thesis_class, top5_predictions):
    thesis_class = normalize_label(thesis_class)

    top5_labels = {
        normalize_label(pred["label"])
        for pred in top5_predictions
    }

    allowed = SEMANTIC_MAP.get(thesis_class, set())

    return len(top5_labels.intersection(allowed)) > 0