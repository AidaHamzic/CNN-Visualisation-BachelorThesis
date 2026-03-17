from torchvision.models import resnet18, ResNet18_Weights


def load_pretrained_resnet18():
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.eval()
    imagenet_classes = weights.meta["categories"]
    return model, imagenet_classes