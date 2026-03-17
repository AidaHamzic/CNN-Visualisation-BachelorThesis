from torchvision.models import vgg16, VGG16_Weights


def load_pretrained_vgg16():
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights=weights)
    model.eval()
    imagenet_classes = weights.meta["categories"]
    return model, imagenet_classes