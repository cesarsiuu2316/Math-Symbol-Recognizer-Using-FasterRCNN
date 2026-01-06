import os
import cv2
import numpy as np
import json
import random
from tqdm import tqdm
import math
from utils import load_config

def load_and_clean_source_data(img_dir, annotations_file):
    """
    1. Loads CROHME images.
    2. Binarizes them (Black Ink, White BG).
    3. Crops them tightly to remove original padding.
    Args:
        img_dir: Directory containing CROHME images.
        annotations_file: JSON file with annotations.
    Returns: 
        List of 'Stamps' (dictionaries with img, boxes, labels).
    """
    if not os.path.exists(annotations_file):
        print("Source annotations not found.")
        return []

    with open(annotations_file, 'r') as f:
        original_annotations = json.load(f)
        
    stamps = []
    print("Pre-processing source stamps...")
    
    image_data = original_annotations['annotations']
    
    for item in tqdm(image_data):
        img_path = os.path.join(img_dir, item['image_name'])
        if not os.path.exists(img_path): continue
        
        # Load grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            print(f"Warning: Could not load image {img_path}, skipping.")
            continue
        
        original_boxes = np.array(item['boxes'])
        if len(original_boxes) == 0: 
            print(f"Warning: No bounding boxes found for image {img_path}, skipping.")
            continue
        
        # Find the extent of all symbols in the image
        min_x = np.min(original_boxes[:, 0])
        min_y = np.min(original_boxes[:, 1])
        max_x = np.max(original_boxes[:, 2])
        max_y = np.max(original_boxes[:, 3])
        
        # Add padding
        pad = 5
        crop_x1 = int(max(0, min_x - pad))
        crop_y1 = int(max(0, min_y - pad))
        crop_x2 = int(min(img.shape[1], max_x + pad))
        crop_y2 = int(min(img.shape[0], max_y + pad))
        
        # Crop the image
        # If crop is invalid (width or height is 0), skip
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1: 
            print("Invalid crop dimensions, skipping image.")
            continue
        
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Adjust boxes to be relative to the new crop
        new_boxes = original_boxes.copy()
        new_boxes[:, 0] -= crop_x1
        new_boxes[:, 2] -= crop_x1
        new_boxes[:, 1] -= crop_y1
        new_boxes[:, 3] -= crop_y1
        
        stamps.append({
            'img': crop,
            'boxes': new_boxes.tolist(),
            'labels': item['labels']
        })
            
    print(f"Loaded {len(stamps)} usable stamps.")
    return stamps

def generate_mosaic(stamps, stamp_deck, canvas_size, min_symbols_per_image, max_symbols_per_image, max_attempts_per_symbol, overlap_tolerance, scaling_factor):
    """
    Places random stamps onto a blank canvas to create a synthetic image.
    Args:
        stamps: List of stamp dictionaries with 'img', 'boxes', 'labels'.
        stamp_deck: List of indices to select stamps from.
        canvas_size: Tuple (H, W)
        min_symbols_per_image: Minimum number of symbols to place.
        max_symbols_per_image: Maximum number of symbols to place.
        max_attempts_per_symbol: Max attempts to place a symbol without overlap.
        scaling_factor: Factor to scale each stamp.
    Returns:
        canvas: The generated synthetic image (numpy array).
        final_boxes: List of bounding boxes for all placed symbols.
        final_labels: List of labels corresponding to the boxes.
    """
    H, W = canvas_size
    canvas = np.ones((H, W), dtype=np.uint8) * 255 # White Canvas
    
    final_boxes = []
    final_labels = []
    
    # Track occupied space to avoid overlapping equations
    # Store as [x1, y1, x2, y2] of the whole stamp
    occupied_rects = []
    
    num_items = random.randint(min_symbols_per_image, max_symbols_per_image)

    count_bigger = 0

    for _ in range(num_items):
        # Refill deck if empty
        if len(stamp_deck) == 0:
            stamp_deck.extend(list(range(len(stamps))))
            random.shuffle(stamp_deck)
            
        # Pop next index
        stamp_idx = stamp_deck.pop()
        stamp = stamps[stamp_idx]
        
        stamp_boxes = np.array(stamp['boxes'])
        
        if len(stamp_boxes) == 0: 
            continue
        
        # Resize the stamp image
        s_h, s_w = stamp['img'].shape
        new_w = int(s_w * scaling_factor)
        new_h = int(s_h * scaling_factor)
        
        # Don't place if it's bigger than the canvas
        if new_w >= W or new_h >= H: 
            print("Stamp too large for canvas after scaling, skipping.")
            count_bigger += 1
            continue
        
        resized_img = cv2.resize(stamp['img'], (new_w, new_h))
        resized_boxes = stamp_boxes * scaling_factor
        
        # --- Random Placement with Collision Detection ---
        for _ in range(max_attempts_per_symbol):
            x_pos = random.randint(0, W - new_w)
            y_pos = random.randint(0, H - new_h)
            
            # Define candidate area for the new stamp
            candidate_rect = [x_pos, y_pos, x_pos + new_w, y_pos + new_h]
            
            collision = False
            for r in occupied_rects:
                # r is [x1, y1, x2, y2]
                r_w = r[2] - r[0]
                r_h = r[3] - r[1]
                
                # Shrink the "Hit Box" of the existing rect
                # New stamp can overlap the margin, but not the 'inner_r'
                shrink_x = r_w * overlap_tolerance
                shrink_y = r_h * overlap_tolerance
                
                inner_r = [
                    r[0] + shrink_x, 
                    r[1] + shrink_y, 
                    r[2] - shrink_x, 
                    r[3] - shrink_y
                ]
                
                # Check intersection against the shrunken rectangle
                if (candidate_rect[0] < inner_r[2] and candidate_rect[2] > inner_r[0] and
                    candidate_rect[1] < inner_r[3] and candidate_rect[3] > inner_r[1]):
                    collision = True
                    break
            
            if not collision:
                # --- Paste the image ---
                target_area = canvas[y_pos:y_pos+new_h, x_pos:x_pos+new_w]
                canvas[y_pos:y_pos+new_h, x_pos:x_pos+new_w] = np.minimum(target_area, resized_img)
                
                # Shift boxes global coordinates
                shifted_boxes = resized_boxes.copy()
                shifted_boxes[:, 0] += x_pos
                shifted_boxes[:, 2] += x_pos
                shifted_boxes[:, 1] += y_pos
                shifted_boxes[:, 3] += y_pos
                
                final_boxes.extend(shifted_boxes.tolist())
                final_labels.extend(stamp['labels'])
                
                occupied_rects.append(candidate_rect)
                break
        print(f"Skipped {count_bigger} stamps that were too big for the canvas.")
                
    return canvas, final_boxes, final_labels

def main():
    # Load configs
    if len(os.sys.argv) < 2:
        print("Usage:")
        print(f"\tpython {os.sys.argv[0]} <path_to_config.json>")
        exit(1)
    config = load_config(os.sys.argv[1])

    # Paths to original CROHME data
    original_img_dir = config['paths']['original_image_dir']
    original_annotations_path = config['paths']['original_annotations_path']    
    # Paths for synthetic data
    synthetic_image_dir = config['paths']['synthetic_image_dir']
    synthetic_annotations_path = config['paths']['synthetic_annotations_path']
    
    # Synthetic Image Params
    synthetic_image_params = config['synthetic_data_params']
    canvas_size = tuple(synthetic_image_params['canvas_size']) # (H, W)
    num_synthetic_images = synthetic_image_params['num_synthetic_images']
    min_symbols_per_image = synthetic_image_params['min_symbols_per_image']
    max_symbols_per_image = synthetic_image_params['max_symbols_per_image']
    max_attempts_per_symbol = synthetic_image_params['max_attempts_per_symbol']
    overlap_tolerance = synthetic_image_params['overlap_tolerance']
    scaling_factor = config['transform_params']['scaling_factor']
    # ---------------------
    
    os.makedirs(synthetic_image_dir, exist_ok=True)
    
    # 1. Load Source
    stamps = load_and_clean_source_data(original_img_dir, original_annotations_path)
    
    # 2. Init Deck (List of indices 0 to N)
    stamp_deck = list(range(len(stamps)))
    random.shuffle(stamp_deck)
    
    # 3. Generate
    all_annotations = []
    print(f"Generating {num_synthetic_images} synthetic images...")

    for i in tqdm(range(num_synthetic_images)):
        canvas, boxes, labels = generate_mosaic(
            stamps=stamps,
            stamp_deck=stamp_deck,
            canvas_size=canvas_size,
            min_symbols_per_image=min_symbols_per_image,
            max_symbols_per_image=max_symbols_per_image,
            max_attempts_per_symbol=max_attempts_per_symbol,
            overlap_tolerance=overlap_tolerance,
            scaling_factor=scaling_factor            
        )
        
        filename = f"syn_{i:05d}.png"
        cv2.imwrite(os.path.join(synthetic_image_dir, filename), canvas)
        
        all_annotations.append({
            "image_name": filename,
            "width": canvas_size[1],
            "height": canvas_size[0],
            "boxes": boxes,
            "labels": labels
        })
        
    # 3. Save JSON
    final_json = {
        "source:" : "Synthetic dataset generated from CROHME stamps",
        "annotations": all_annotations
        }
    with open(synthetic_annotations_path, 'w') as f:
        json.dump(final_json, f, indent=4)
        
    print(f"Done. Saved to {synthetic_annotations_path}")

if __name__ == "__main__":
    main()