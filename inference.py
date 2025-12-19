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
    
    # 1. Build Architecture
    print("Building model...")
    model = get_model(config)
    
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