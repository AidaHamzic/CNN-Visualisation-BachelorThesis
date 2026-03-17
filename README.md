# CNN Visualisation – Bachelor Thesis

This thesis implements a comparative visual analysis of convolutional neural networks (CNNs) for image classification.

The goal is to analyse how different pretrained models (VGG16, ResNet18, MobileNetV2) make decisions on the same dataset, and to visualise their internal behaviour using Grad-CAM and feature map extraction.

The pipeline consists of:
1. Running inference on the Places365 dataset (~36,500 images)
2. Extracting predictions and top-5 probabilities
3. Mapping predictions to thesis-defined semantic classes
4. Identifying:
   - Correct classifications
   - Failure cases
   - Cross-model disagreements
5. Visualising model behaviour via:
   - Grad-CAM heatmaps
   - Feature maps (early, middle, late layers)
6. Interactive exploration using Streamlit


## Models
- VGG16 (pretrained on ImageNet)
- ResNet18 (pretrained on ImageNet)
- MobileNetV2 (pretrained on ImageNet)
All models are evaluated on the same dataset and aligned using semantic mapping to ensure comparability.

## Dataset

- Places365 (validation subset)
- ~36,500 images
- Images resized to 224x224
- Normalisation using ImageNet statistics

## Key Components
### Manifest Generation
Scripts:
- `build_vgg16_manifest.py`
- `build_resnet18_manifest.py`

Each manifest contains:
- image_name
- image_path
- true_scene_label
- predicted_label
- confidence
- top5_predictions
- thesis_class
- semantically_aligned

### Subset Filtering
Scripts:
- `filter_vgg16_subset.py`
- `filter_resnet18_subset.py`

Generates:
- Correct classification subset
- Failure cases subset
- 
### Semantic Mapping

- `semantic_mapping.py`
- `places_mapping.py`

Used to:
- Map dataset labels to thesis classes
- Align ImageNet predictions with dataset semantics
  
### Visualisation

- Grad-CAM generation
- Feature map extraction (layer-wise)
- Stored in:
  - `outputs/<model>_gradcam/`
  - `outputs/<model>_feature_maps/`



### Streamlit Interface

Run:

```bash
streamlit run interface/streamlit_interface.py
