import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json
from utils import load_config
import torch
import os

def get_model(config):
    """
    Constructs the Faster R-CNN model with custom anchors and class head.
    Args:
        config (dict): Configuration dictionary containing 'model_params' and 'paths'.
        
    Returns:
        model (torch.nn.Module): The configured Faster R-CNN model.
    """
    
    # 1. Load Class Mapping to determine num_classes
    class_mapping_path = config['paths']['class_mapping_path']
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    num_classes = len(class_mapping) + 1 # background class is 0
    
    # 2. Configure Custom Anchor sizes and aspect ratios found in EDA
    anchor_sizes = config['model_params']['anchor_params']['sizes']
    aspect_ratios = config['model_params']['anchor_params']['aspect_ratios']
    
    # AnchorGenerator expects a tuple of tuples for sizes and aspect ratios
    # One tuple for each feature map level. We use the same for all levels here for simplicity.
    # In a standard FPN with 5 levels, we repeat the configuration 5 times.
    anchor_generator = AnchorGenerator(
        sizes=tuple([tuple(anchor_sizes) for _ in range(5)]),
        aspect_ratios=tuple([tuple(aspect_ratios) for _ in range(5)])
    )
    
    # 3. Load Pre-trained Model
    # We start with a ResNet50-FPN backbone pre-trained on COCO.
    # weights="DEFAULT" loads the best available weights.
    print(f"Loading Faster R-CNN with backbone: {config['model_params']['weights_backbone']}")
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=config['model_params']['weights'], # Usually None if we want to start fresh or specific weights
        weights_backbone=config['model_params']['weights_backbone'], # ImageNet weights for backbone
        trainable_backbone_layers=config['model_params']['trainable_backbone_layers'], # Unfreeze all layers (5) for domain adaptation
        rpn_anchor_generator=anchor_generator
    )
    
    # 4. Replace the Head (Classifier)
    # The pre-trained model has 91 classes (COCO). We need to replace the predictor
    # with one that has our specific number of math symbol classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 5. Configure ROI Heads (Inference parameters)
    # Adjust NMS and Score thresholds based on config
    model.roi_heads.score_thresh = config['model_params']['roi_heads']['box_score_thresh']
    model.roi_heads.nms_thresh = config['model_params']['roi_heads']['box_nms_thresh']
    model.roi_heads.detections_per_img = config['model_params']['roi_heads']['box_detections_per_img']
    
    return model

def testing(config):    
    print("Building model...")
    model = get_model(config)
    print("Model built successfully.")
    
    # Print model summary or check a specific layer
    print("\nModel Summary:")
    print("----------------")
    print(model)
    print(f"Number of classes: {model.roi_heads.box_predictor.cls_score.out_features}")
    print(f"Anchor Generator sizes: {model.rpn.anchor_generator.sizes}")
    print(f"Anchor Generator aspect ratios: {model.rpn.anchor_generator.aspect_ratios}")
    
    # Test with a dummy input
    print("\nTesting forward pass with dummy input...")
    model.eval()
    x = [torch.rand(3, 320, 320)]
    predictions = model(x)
    print("Forward pass successful.")
    print(f"Output keys: {predictions[0].keys()}")

def main():
    # Load config
    if len(os.sys.argv) < 2:
        print("Usage:")
        print(f"\tpython {os.sys.argv[0]} <path_to_config.json>")
        exit(1)
    config = load_config(os.sys.argv[1])

    testing(config)

if __name__ == "__main__":
    main()