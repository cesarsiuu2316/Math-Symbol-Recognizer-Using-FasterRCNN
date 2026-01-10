import os
import cv2
import numpy as np
import json
import random
import torch
from tqdm import tqdm
import math
from utils import load_config
import albumentations as A

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
        if not os.path.exists(img_path): 
            print(f"Warning: Image {img_path} not found, skipping.")
            continue
        
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

def transform_stamp(stamp, transform_params, class_mapping):
    """
    Applies transformations to a stamp image and its boxes.
    Currently a placeholder for future augmentations.
    Args:
        stamp: Dictionary with 'img', 'boxes', 'labels'.
        transform_params: Parameters for transformations loaded from config.
    Returns:
        transformed_img: Transformed image (numpy array).
        transformed_boxes: Transformed bounding boxes (numpy array).
        labels: Labels (list).
    """
    # transformation parameters
    transform = transform_params['transform']
    scaling_factor = transform_params['scaling_factor']
    target_min_size = transform_params['target_min_size']
    target_max_size = transform_params['target_max_size']
    # Augmentation parameters
    augmentation_params = transform_params['augmentation_params']
    morphological_ops = augmentation_params['morphological_ops']
    morphological_kernels = augmentation_params['morphological_kernels']
    # Get affine configs
    affine_rotate = augmentation_params['affine_rotate']
    affine_shear = augmentation_params['affine_shear']
    affine_fill_value = augmentation_params['affine_fill_value']
    affine_probability = augmentation_params['affine_probability']

    img = stamp['img']
    boxes = np.array(stamp['boxes'], dtype=np.float32)
    labels_original = stamp['labels']
    labels = torch.tensor([class_mapping[l] for l in stamp['labels']], dtype=torch.int64)

    # ----------- 1. Dynamic Scaling (Min size not needed right now) ---------------
    h, w = img.shape
    new_w = int(w * scaling_factor)
    new_h = int(h * scaling_factor)

    # Scale down if larger than target_max_size
    max_dim = max(new_w, new_h)
    if max_dim > target_max_size:
        scale_down_factor = target_max_size / max_dim
        new_w = int(new_w * scale_down_factor)
        new_h = int(new_h * scale_down_factor)
    
    # resize image
    img = cv2.resize(img, (new_w, new_h))
    
    # Scale boxes: [x1, y1, x2, y2]
    if len(boxes) > 0:
        boxes[:, [0, 2]] *= (new_w / w)
        boxes[:, [1, 3]] *= (new_h / h)

    # ------------------------ AFFINE AND MORPHOLOGICAL TRANFORMATIONS ------------------------
    if transform:
        # change img to rgb for albumentations
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # ---------------------- 2. Affine (Rotation/Shear) -------------------------------
        if random.random() < affine_probability:
            # 1. Calculate Rotation-Safe Padding based on WIDTH
            # sin(10 degrees) approx 0.17. We use 0.2 for safety.
            current_h, current_w = img_rgb.shape[:2]
            
            # Add padding to Top and Bottom based on the WIDTH
            rot_pad_h = int(current_w * 0.2) 
            # Add a little to Left/Right just in case of shear/shift based on height
            rot_pad_w = int(current_h * 0.2) 

            top_pad = rot_pad_h // 2
            bottom_pad = rot_pad_h - top_pad
            left_pad = rot_pad_w // 2
            right_pad = rot_pad_w - left_pad
            # Apply the padding using OpenCV
            img_rgb = cv2.copyMakeBorder(
                img_rgb, 
                top=top_pad, bottom=bottom_pad, 
                left=left_pad, right=right_pad,
                borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )

            # Shift boxes to match new coordinates
            if len(boxes) > 0:
                boxes[:, 0] += left_pad
                boxes[:, 2] += left_pad
                boxes[:, 1] += top_pad
                boxes[:, 3] += top_pad

            # Clip boxes to image size before affine transform
            if len(boxes) > 0:
                h_aug, w_aug = img_rgb.shape[:2]
                boxes[:, 0] = np.clip(boxes[:, 0], 0, w_aug)
                boxes[:, 1] = np.clip(boxes[:, 1], 0, h_aug)
                boxes[:, 2] = np.clip(boxes[:, 2], 0, w_aug)
                boxes[:, 3] = np.clip(boxes[:, 3], 0, h_aug)   

            
            # Affine transformations
            affine_transform = A.Compose([
                A.Affine(
                    rotate=tuple(affine_rotate),
                    shear=tuple(affine_shear),
                    fill=affine_fill_value,
                    p=1.0
                )
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
            # pascal voc format: [x_min, y_min, x_max, y_max]         
            
            try:
                transformed = affine_transform(
                    image=img_rgb, 
                    bboxes=boxes.tolist(), 
                    labels=labels.tolist()
                )
                img_rgb = transformed['image']
                boxes = np.array(transformed['bboxes'], dtype=np.float32)

                # Clip boxes to image size after affine transform due to possible rotation/translation
                if len(boxes) > 0:
                    h_aug, w_aug = img_rgb.shape[:2]
                    boxes[:, 0] = np.clip(boxes[:, 0], 0, w_aug)
                    boxes[:, 1] = np.clip(boxes[:, 1], 0, h_aug)
                    boxes[:, 2] = np.clip(boxes[:, 2], 0, w_aug)
                    boxes[:, 3] = np.clip(boxes[:, 3], 0, h_aug)  
            except Exception as e:
                print(f"Error in affine transformation for image: {e}")
                pass # keep original if error occurs

        # Convert back to grayscale
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # ------------------------- 3. Morphology (Thickness) -----------------------------
        # Probability of one of two morphological ops = 0.3 * 0.5 = 0.15
        if random.random() < 0.3:
            morpho_kernel_size = random.choice(morphological_kernels)
            kernel = np.ones((morpho_kernel_size, morpho_kernel_size), np.uint8)
            op = random.choice(morphological_ops)
            # print(f"OP: {op}, Kernel Size: {morpho_kernel_size}")
            if op == 'dilate':
                # Dilate image = Max filter = Expands White (Ink gets Thinner)
                img_gray = cv2.dilate(img_gray, kernel, iterations=1)
            elif op == 'erode':
                # Erode image = Min filter = Expands Black (Ink gets Thicker)
                img_gray = cv2.erode(img_gray, kernel, iterations=1)

    return img_gray, boxes, labels_original



def generate_mosaic(stamps, stamp_deck, canvas_size, min_symbols_per_image, max_symbols_per_image, max_attempts_per_symbol, overlap_tolerance, transform_params, class_mapping):
    """
    Places random stamps onto a blank canvas to create a synthetic image.
    Args:
        stamps: List of stamp dictionaries with 'img', 'boxes', 'labels'.
        stamp_deck: List of indices to select stamps from.
        canvas_size: Tuple (H, W)
        min_symbols_per_image: Minimum number of symbols to place.
        max_symbols_per_image: Maximum number of symbols to place.
        max_attempts_per_symbol: Max attempts to place a symbol without overlap.
        overlap_tolerance: Fractional tolerance for overlap detection.
        transform_params: Parameters for transformations loaded from config.
    Returns:
        canvas: The generated synthetic image (numpy array).
        final_boxes: List of bounding boxes for all placed symbols.
        final_labels: List of labels corresponding to the boxes.
    """
    H, W = canvas_size
    canvas = np.ones((H, W), dtype=np.uint8) * 255 # White Canvas
    
    final_boxes = []
    final_labels = []
    occupied_rects = []
    
    num_items = random.randint(min_symbols_per_image, max_symbols_per_image)
    count_skipped = 0

    for _ in range(num_items):
        # Refill deck if empty
        if len(stamp_deck) == 0:
            stamp_deck.extend(list(range(len(stamps))))
            random.shuffle(stamp_deck)
            
        # Pop next index
        stamp_idx = stamp_deck.pop()
        stamp = stamps[stamp_idx]
        
        transformed_img, transformed_boxes, labels = transform_stamp(stamp, transform_params, class_mapping)
        
        h_stamp, w_stamp = transformed_img.shape

        # Don't place if it's bigger than the canvas
        if w_stamp >= W or h_stamp >= H: 
            print("Stamp too large for canvas after scaling, skipping.")
            count_skipped += 1
            continue
        
        # --- Random Placement with Collision Detection ---
        placed = False
        for _ in range(max_attempts_per_symbol):
            x_pos = random.randint(0, W - w_stamp)
            y_pos = random.randint(0, H - h_stamp)
            
            # Define candidate area for the new stamp
            candidate_rect = [x_pos, y_pos, x_pos + w_stamp, y_pos + h_stamp]
            
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
                target_area = canvas[y_pos:y_pos+h_stamp, x_pos:x_pos+w_stamp]
                canvas[y_pos:y_pos+h_stamp, x_pos:x_pos+w_stamp] = np.minimum(target_area, transformed_img)
                
                # Shift boxes global coordinates
                shifted_boxes = transformed_boxes.copy()
                shifted_boxes[:, 0] += x_pos
                shifted_boxes[:, 2] += x_pos
                shifted_boxes[:, 1] += y_pos
                shifted_boxes[:, 3] += y_pos
                
                final_boxes.extend(shifted_boxes.tolist())
                final_labels.extend(labels)                
                occupied_rects.append(candidate_rect)
                placed = True
                break

        if not placed: 
            count_skipped += 1
            #print("Could not place stamp without collision, skipping and adding to deck again.")
            # add back the stamp to the deck for future use
            stamp_deck.append(stamp_idx)
                    
    return canvas, final_boxes, final_labels

def main():
    # Load configs
    if len(os.sys.argv) < 2:
        print("Usage:")
        print(f"\tpython {os.sys.argv[0]} <path_to_config.json>")
        exit(1)
    config = load_config(os.sys.argv[1])

    # Load class mapping to ensure correct IDs
    class_mapping_path = config['paths']['class_mapping_path']
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)

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
    transform_params = config['transform_params']
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
            transform_params=transform_params,
            class_mapping=class_mapping       
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