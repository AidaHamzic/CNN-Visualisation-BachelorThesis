from models.vgg16_feature_maps import generate_vgg16_feature_maps
from models.vgg16_gradcam import generate_vgg16_gradcam

from models.resnet18_feature_maps import generate_resnet18_feature_maps
from models.resnet18_gradcam import generate_resnet18_gradcam


MODEL_CONFIG = {
    "vgg16": {
        "display_name": "VGG16",
        "feature_maps_fn": generate_vgg16_feature_maps,
        "gradcam_fn": generate_vgg16_gradcam,
        "correct_csv": "outputs/vgg16_correct_subset.csv",
        "failure_csv": "outputs/vgg16_failure_subset.csv",
        "subset_csv": "outputs/vgg16_places365_subset_manifest.csv",
    },
    "resnet18": {
        "display_name": "ResNet-18",
        "feature_maps_fn": generate_resnet18_feature_maps,
        "gradcam_fn": generate_resnet18_gradcam,
        "correct_csv": "outputs/resnet18_correct_subset.csv",
        "failure_csv": "outputs/resnet18_failure_subset.csv",
        "subset_csv": "outputs/resnet18_places365_subset_manifest.csv",
    },
}