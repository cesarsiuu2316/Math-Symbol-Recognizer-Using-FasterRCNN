import torch
import cv2
import os
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog
from utils import load_config
from model import get_model

def load_trained_model(config, model_path=None):
    """
    Loads the config and the trained model weights.
    """
    device = torch.device(config['training_params']['device'])

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
    
    # 2. Load Weights
    if model_path:
        pass
    else:
        model_path = config['paths']['final_model_path']

    #model_path = "output/models/fasterrcnn_resnet50_fpn.chkpt_optim_32.dat"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
        
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval() # Set to evaluation mode (no gradients, inference behavior)
    
    return model, device

def get_class_names(config):
    """
    Loads the class mapping and inverts it (ID -> Name).
    """
    mapping_path = config['paths']['class_mapping_path']
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Invert mapping: {"alpha": 1} -> {1: "alpha"}
    # Note: Model Output 0 is always Background.
    id_to_name = {v: k for k, v in class_mapping.items()}
    return id_to_name

def select_image():
    """
    Opens a file dialog to select an image.
    """
    root = tk.Tk()
    root.withdraw() # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select an Image for Inference",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    return file_path

def predict_and_draw(model, device, image_path, id_to_name, threshold=0.5):
    """
    Runs inference and draws bounding boxes.
    """
    # 1. Load Image
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- NEW: PREVENT RESIZING TRICK ---
    # We get the actual dimensions of the loaded image
    h, w = img_rgb.shape[:2]
    min_dim = min(h, w)
    max_dim = max(h, w)

    # We force the model's transformation parameters to match this specific image.
    # Logic: scale = min_size / min_dim
    # If min_size == min_dim, then scale == 1.0 (No Resize)
    model.transform.min_size = (min_dim,) 
    model.transform.max_size = max_dim 
    # -----------------------------------
    
    # 2. Preprocess (Normalize and Convert to Tensor)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dimension [1, C, H, W]
    # 3. Inference
    print("Running inference...")
    with torch.no_grad():
        prediction = model(img_tensor)[0] # Get first (and only) image result
        
    # 4. Filter & Draw
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    img_result = img_bgr.copy()
    
    print(f"Found {len(scores)} raw detections.")
    
    count = 0
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            count += 1
            x1, y1, x2, y2 = box.astype(int)
            class_name = id_to_name.get(label, f"Unknown ({label})")
            label = f"{class_name}: {score:.2f}"
            
            # Draw Box (Red)
            # Color varies according to score percentage
            # Greener > 95 to Yellow > 90 to Redder > 80
            color = (0, int(255 * min((score - 0.8) / 0.15, 1.0)), int(255 * max(0, 1.0 - (score - 0.8) / 0.15)))
            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label (White text with Red background)
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_result, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(img_result, class_name, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    print(f"Displayed {count} detections with score > {threshold}")
    return img_result

def main():
    # 1. Load Config and Model
    model_path = None
    if len(os.sys.argv) < 2:
        print(f"Usage: python {os.sys.argv[0]} <config.json>")
        print(f"\tOptionally: python {os.sys.argv[0]} <config.json> <checkpoint_path>")
        os.sys.exit(1)
    elif len(os.sys.argv) == 3:
        print("Loading checkpoint...")
        model_path = os.sys.argv[2]
        if not os.path.exists(model_path):
            print(f"Error: Checkpoint file not found at {model_path}")
            os.sys.exit(1)
    
    config = load_config(os.sys.argv[1])
    try:
        model, device = load_trained_model(config, model_path)
        id_to_name = get_class_names(config)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Select Image
    while True:
        img_path = select_image()
        if not img_path:
            print("No image selected.")
            return

        # 3. Run Inference
        # Threshold: Only show boxes with > 50% confidence
        result_img = predict_and_draw(model, device, img_path, id_to_name, threshold=0.7)
        
        # 4. Save and Show Results
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, f"result_{filename}")
        
        cv2.imwrite(save_path, result_img)
        print(f"Result saved to: {save_path}")
        
        cv2.imshow(f"Inference Result - {filename}", result_img)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Do you want to process another image? (y/n): ", end="")
        choice = input().strip().lower()
        if choice != 'y':
            break

if __name__ == "__main__":
    main()