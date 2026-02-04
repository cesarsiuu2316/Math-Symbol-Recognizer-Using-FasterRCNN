import os
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json

from utils import load_config

def get_model(num_classes, anchor_sizes, aspect_ratios, weights, weights_backbone, trainable_backbone_layers, num_fpn_levels, skip_resize, 
        min_size, max_size, score_thresh, nms_thresh, detections_per_img):
    """
    Constructs the Faster R-CNN model with custom anchors and class head.
    Args:
        num_classes (int): Number of classes including background.
        anchor_sizes (list of int): Anchor sizes for RPN.
        aspect_ratios (list of float): Aspect ratios for RPN.
        weights: Pre-trained weights for the model backbone.
        weights_backbone: Pre-trained weights for the backbone.
        trainable_backbone_layers (int): Number of trainable layers in the backbone.
        num_fpn_levels (int): Number of FPN levels.
        skip_resize (bool): Whether to skip resizing in the model transform.
        min_size (int): Minimum size for image resizing.
        max_size (int): Maximum size for image resizing.
        score_thresh (float): Score threshold for ROI heads.
        nms_thresh (float): NMS threshold for ROI heads.
        detections_per_img (int): Max detections per image for ROI heads.
        
    Returns:
        model (torch.nn.Module): The configured Faster R-CNN model.
    """
    
    # AnchorGenerator expects a tuple of tuples for sizes and aspect ratios
    # One tuple for each feature map level. We use the same for all levels here for simplicity.
    # In a standard FPN with 5 levels, we repeat the configuration 5 times.
    anchor_generator = AnchorGenerator(
        sizes=tuple([tuple(anchor_sizes) for _ in range(num_fpn_levels)]),
        aspect_ratios=tuple([tuple(aspect_ratios) for _ in range(num_fpn_levels)])
    )
    
    # 3. Load Pre-trained Model
    # We start with a ResNet50-FPN backbone pre-trained on COCO.
    # weights="DEFAULT" loads the best available weights.
    print(f"Loading Faster R-CNN with backbone: {weights_backbone}")
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights, # Usually None if we want to start fresh or specific weights
        weights_backbone=weights_backbone, # ImageNet weights for backbone
        trainable_backbone_layers=trainable_backbone_layers, # Unfreeze all layers (5) for domain adaptation
        rpn_anchor_generator=anchor_generator, # Custom Anchor Generator for all FPN levels
        _skip_resize=skip_resize, # Whether to skip resizing in the model transform
        min_size=min_size, # Not really used if skip_resize is True
        max_size=max_size # Not really used if skip_resize is True
    )
    
    # 4. Replace the Head (Classifier)
    # The pre-trained model has 91 classes (COCO). We need to replace the predictor to match our number of classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 5. Configure ROI Heads (Inference parameters)
    # Adjust NMS and Score thresholds based on config
    model.roi_heads.score_thresh = score_thresh
    model.roi_heads.nms_thresh = nms_thresh
    model.roi_heads.detections_per_img = detections_per_img
    
    return model

if __name__ == "__main__":
    if len(os.sys.argv) < 2:
        print("Usage:")
        print(f"\tpython {os.sys.argv[0]} <path_to_config.json>")
        exit(1)

    config = load_config(os.sys.argv[1])
    class_mapping_path = config['paths']['class_mapping_path']
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    num_classes = len(class_mapping) + 1 # background class is 0
    anchor_sizes = config['model_params']['anchor_params']['sizes'] # Custom anchor sizes
    aspect_ratios = config['model_params']['anchor_params']['aspect_ratios'] # Custom aspect ratios
    weights = config['model_params']['weights'] # Usually None if we want to start fresh or specific weights
    weights_backbone = config['model_params']['weights_backbone'] # ImageNet weights for backbone
    trainable_backbone_layers = config['model_params']['trainable_backbone_layers'] # Unfreeze all layers (5) for domain adaptation
    num_fpn_levels = config['model_params']['num_fpn_levels'] # Number of FPN levels
    skip_resize = config['model_params']['skip_resize'] # Whether to skip resizing in the model
    min_size = config['transform_params']['target_min_size'] # Min size for resizing (not used, but still passed)
    max_size = config['model_params']['target_max_size'] # Max size for resizing (not used, but still passed)
    score_thresh = config['model_params']['roi_heads']['box_score_thresh'] # Score threshold for ROI heads
    nms_thresh = config['model_params']['roi_heads']['box_nms_thresh'] # NMS threshold for ROI heads
    detections_per_img = config['model_params']['roi_heads']['box_detections_per_img'] # Max detections per image

    model = get_model(
        num_classes=num_classes,
        anchor_sizes=anchor_sizes, 
        aspect_ratios=aspect_ratios, 
        weights=weights, 
        weights_backbone=weights_backbone, 
        trainable_backbone_layers=trainable_backbone_layers, 
        num_fpn_levels=num_fpn_levels,
        skip_resize=skip_resize, 
        min_size=min_size, 
        max_size=max_size, 
        score_thresh=score_thresh, 
        nms_thresh=nms_thresh, 
        detections_per_img=detections_per_img
    )

    print(model)