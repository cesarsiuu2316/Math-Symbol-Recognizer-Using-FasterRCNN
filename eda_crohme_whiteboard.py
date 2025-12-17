import json
import os
import cv2
import numpy as np
import glob

def load_state(calibrated_whiteboard_bboxes_file):
    """Internal helper to load calibration progress."""
    if os.path.exists(calibrated_whiteboard_bboxes_file):
        with open(calibrated_whiteboard_bboxes_file, 'r') as f:
            return json.load(f)
    return {}

def save_state(data, calibrated_whiteboard_bboxes_file):
    """Internal helper to save calibration progress."""
    with open(calibrated_whiteboard_bboxes_file, 'w') as f:
        json.dump(data, f, indent=4)

def calculate_optimal_max_size(annotations_path, scaling_factor, percentile=95):
    """
    Calculates the optimal target_max_size based on the distribution of 
    CROHME images after applying the whiteboard scaling factor.
    Args: 
        annotations_path (str): Path to the CROHME annotations JSON file.
        scaling_factor (float): Scaling factor derived from whiteboard calibration.
        percentile (int): Percentile to determine optimal max_size.
    Returns:
        int: Calculated optimal target_max_size.
    """
    if not os.path.exists(annotations_path):
        print(f"Error: Annotations file not found at {annotations_path}")
        return None

    with open(annotations_path, 'r') as f:
        data = json.load(f)
        
    scaled_dimensions = []
    
    print(f"Calculating optimal max_size (p{percentile}) with factor {scaling_factor:.4f}...")
    
    for item in data['annotations']:
        # Original dims
        w = item['width']
        h = item['height']
        
        # Apply Scaling
        new_w = w * scaling_factor
        new_h = h * scaling_factor
        
        # We care about the maximum dimension of the image
        scaled_dimensions.append(max(new_w, new_h))
        
    if not scaled_dimensions:
        return 1333 # Default fallback
        
    # Calculate Percentile
    optimal_max_size = np.percentile(np.array(scaled_dimensions), percentile)
    max_absolute = np.max(scaled_dimensions)
    id_max = np.argmax(scaled_dimensions)
    
    print(f"-> Distribution Stats (Scaled):")
    print(f"   Max Dimension (Absolute): {int(max_absolute)} px")
    print(f"   Max Dimension (p{percentile}): {int(optimal_max_size)} px")
    print(f"   Image causing max dimension: {data['annotations'][id_max]['image_name']}")
    return int(optimal_max_size)

def calculate_crohme_stats(annotations_path):
    """
    Calculates median width, height, and aspect ratios from CROHME annotations.
    Args:
        annotations_path (str): Path to the annotations JSON file.
        
    Returns:
        dict: Dictionary containing median_width, median_height, and aspect_ratios.
    """
    if not os.path.exists(annotations_path):
        print(f"Error: Annotations file not found at {annotations_path}")
        return None

    with open(annotations_path, 'r') as f:
        data = json.load(f)
        
    widths = []
    heights = []
    aspect_ratios = []
    
    print(f"Analyzing {len(data['annotations'])} samples from CROHME...")
    counter = 0

    for item in data['annotations']:
        for bbox in item['boxes']:
            # bbox format: [x1, y1, x2, y2]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            # Check for valid dimensions
            if w > 0 and h > 0:
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
            else:
                print(f"Warning: Invalid bbox {bbox} in file {item['image_name']}")
            
            counter += 1
                
    stats = {
        "median_width": float(np.median(widths)),
        "median_height": float(np.median(heights)),
        "median_area": float(np.median(np.array(widths) * np.array(heights))),
        "aspect_ratios": aspect_ratios # Return all for clustering if needed, or summary stats
    }
    
    print(f"Analyzing {counter} bounding boxes from CROHME...")
    print(f"CROHME Stats: Median W={stats['median_width']:.2f}, Median H={stats['median_height']:.2f}")
    return stats

def interactive_whiteboard_calibration(whiteboard_dir, calibrated_whiteboard_bboxes_path, reset=False):
    """
    Allows user to manually draw boxes on whiteboard images to estimate target symbol size.

    Args:
        whiteboard_dir (str): Directory containing whiteboard images.
        
    Returns:
        float: Median area of symbols on whiteboard.
    """
    # 1. Gather Images
    image_files = glob.glob(os.path.join(whiteboard_dir, "*.*"))
    if not image_files:
        print(f"No images found in {whiteboard_dir}.")
        return None
    
    # 2. State Management
    if reset and os.path.exists(calibrated_whiteboard_bboxes_path):
        os.remove(calibrated_whiteboard_bboxes_path)
        labeled_data = {}
        print("Calibration state reset.")
    else:
        labeled_data = load_state(calibrated_whiteboard_bboxes_path)

    # Filter out images already processed in the state file
    remaining_files = [f for f in image_files if os.path.basename(f) not in labeled_data]

    # If we have data and no new files, just return the calculation
    if not remaining_files and labeled_data:
        print("All images processed. Returning saved stats.")
        areas = []
        for img_data in labeled_data.values():
            areas.extend(img_data)
        return float(np.median(areas))
    
    print("\n--- Interactive Calibration ---")
    print(f"Total Images: {len(image_files)}")
    print(f"Already Labeled: {len(labeled_data)}")
    print(f"Remaining: {len(remaining_files)}")
    print("INSTRUCTIONS:")
    print("1. Draw box -> Let go of mouse -> Press Space or Enter.")
    print("2. When finished with an image, press ESC to move to next image.")
    
    window_name = "Calibration: Draw a Box then press SPACE"
    
    for i, img_path in enumerate(remaining_files):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: 
            continue
            
        print(f"[{i+1}/{len(remaining_files)}] Processing: {img_name}")

        # It returns a tuple of lists [[x,y,w,h], ...]
        rois = cv2.selectROIs(window_name, img, showCrosshair=True, fromCenter=False, printNotice=True)
        cv2.destroyWindow(window_name)

        # Logic: Empty rois means user hit ESC or Enter without drawing -> Stop/Exit
        if len(rois) == 0:
            print("No boxes selected. Ending calibration...")
            break
            
        current_img_areas = []
        for roi in rois:
            x, y, w, h = roi
            # Filter out accidental tiny clicks
            if w > 2 and h > 2: 
                area = w * h
                current_img_areas.append(float(area))
                print(f"  -> Added box: {w}x{h} (Area: {area})")

        # Save state immediately
        if current_img_areas:
            labeled_data[img_name] = current_img_areas
            save_state(labeled_data, calibrated_whiteboard_bboxes_path)

    cv2.destroyAllWindows()
    
    # Calculate Final Median using ALL data (historical + new)
    all_areas = []
    for img_data in labeled_data.values():
        all_areas.extend(img_data)

    if not all_areas:
        print("No valid areas collected.")
        return None
        
    median_area = float(np.median(all_areas))
    print(f"\nCollected {len(all_areas)} samples.")
    print(f"Median Area: {median_area:.2f}")
    
    return median_area

def get_whiteboard_median_via_cc_labeling(whiteboard_dir, debug=True):
    """
    Calculates the median bounding box area of symbols on a whiteboard.
    Optimized for small symbols (~20px) and noisy environments.
    """
    image_files = glob.glob(os.path.join(whiteboard_dir, "*.*"))
    
    if not image_files:
        print("No images found.")
        return None

    # --- CONFIGURATION ---
    VIEW_WIDTH = 1600         # View for debugging
    MIN_AREA_PX = 5           # Min area to consider a symbol
    MAX_SCREEN_PCT = 0.01     # If a box > 1% of screen, it's noise/diagram/frame
    # ---------------------

    all_areas = []

    print(f"Processing {len(image_files)} images...")

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None: 
            continue

        H, W = img.shape[:2]
        img_area = H * W
        
        # Preprocessing (Background Normalization)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        
        # Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7,7), 0)        
        # Apply Otsu's Thresholding in inverted image
        threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Find Components
        # connectivity=8 checks diagonal pixels too (better for handwriting)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold, connectivity=8)
        
        valid_boxes = []
        
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT] 
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            box_area = w * h

            # --- FILTERS TO GET RID OF NOISE ---
            # 1. Too Small?
            if box_area < MIN_AREA_PX: continue            
            # 2. Too Big? (Diagrams/Frames > 1% of screen)
            elif box_area > (img_area * MAX_SCREEN_PCT): continue
            # 3. Aspect Ratio Check (Removes long skinny lines that span > 20% of the screen width/height)
            elif w > (W * 0.2) or h > (H * 0.2): continue
            
            valid_boxes.append(box_area)
            
            if debug:
                # Draw Green Box with red centroid
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cX, cY = centroids[i]
                cv2.circle(img, (int(cX), int(cY)), 3, (0, 0, 255), -1)

        all_areas.extend(valid_boxes)

        if debug:
            # Scale up for display
            scale = VIEW_WIDTH / W
            new_dim = (VIEW_WIDTH, int(H * scale))
            resized_debug = cv2.resize(img, new_dim)
            cv2.imshow(f"CC Stats Debug - {os.path.basename(img_path)}", resized_debug)
            if cv2.waitKey(0) == 27: # ESC to quit
                cv2.destroyAllWindows()
                break
            cv2.destroyWindow(f"CC Stats Debug - {os.path.basename(img_path)}")

    if debug: 
        cv2.destroyAllWindows()

    if not all_areas:
        print("No valid symbols detected.")
        return 0

    median_val = np.median(all_areas)
    print(f"\n--- CC Stats Calibration ---")
    print(f"Total Components Found: {len(all_areas)}")
    print(f"Median Box Area: {median_val:.2f}")
    
    return median_val