from torchvision.models import VGG16_Weights

weights = VGG16_Weights.IMAGENET1K_V1
imagenet_classes = weights.meta["categories"]

search_terms = [
    "alp",
    "valley",
    "bell cote",
    "mountain tent",
    "unicycle",
    "seashore",
    "lakeside",
    "iceberg",
    "snowfield",
    "traffic light",
    "street sign",
]

print("Checking whether labels exist in ImageNet class list:\n")

for term in search_terms:
    matches = [label for label in imagenet_classes if term.lower() in label.lower()]
    print(f"{term}: {matches}")