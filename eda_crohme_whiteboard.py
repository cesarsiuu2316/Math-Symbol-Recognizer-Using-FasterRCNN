import json
import os
import cv2
import numpy as np
import glob
from utils import load_config, update_config

def calculate_crohme_stats(annotations_path="train_annotations.json"):
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


def interactive_whiteboard_calibration(whiteboard_dir):
    """
    Allows user to manually draw boxes on whiteboard images to estimate target symbol size.

    Args:
        whiteboard_dir (str): Directory containing whiteboard images.
        
    Returns:
        float: Median area of symbols on whiteboard.
    """
    # 1. Gather Images
    image_files = glob.glob(os.path.join(whiteboard_dir, "*.*"))
    # Filter for common image extensions
    
    if not image_files:
        print(f"No images found in {whiteboard_dir}.")
        return None
        
    areas = []
    
    print("\n--- Interactive Calibration ---")
    print("INSTRUCTIONS:")
    print("1. Draw box -> Let go of mouse -> Press Space or Enter.")
    print("2. When finished with an image, press ESC to move to next image.")
    
    window_name = "Calibration: Draw a Box then press SPACE"
    
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        print(f"[{i+1}/{len(image_files)}] Processing: {os.path.basename(img_path)}")

        # It returns a tuple of lists [[x,y,w,h], ...]
        rois = cv2.selectROIs(window_name, img, showCrosshair=True, fromCenter=False)
        
        cv2.destroyWindow(window_name)

        # Logic: If user hit ESC or just pressed Enter without drawing, 
        # rois will be empty (or length 0). We treat this as "Finish".
        if len(rois) == 0:
            print("No boxes selected. Ending calibration...")
            break
            
        for roi in rois:
            x, y, w, h = roi
            # Filter out accidental tiny clicks (e.g., 0x0 or 1x1 pixels)
            if w > 2 and h > 2: 
                area = w * h
                areas.append(area)
                print(f"  -> Added box: {w}x{h} (Area: {area})")

    cv2.destroyAllWindows()
    
    if not areas:
        print("No valid areas collected.")
        return None
        
    median_area = float(np.median(areas))
    print(f"\nCollected {len(areas)} samples.")
    print(f"Median Area: {median_area:.2f}")
    
    return median_area

def get_whiteboard_median_via_cc_labeling(whiteboard_dir, debug=False):
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
    BORDER_MASK_PCT = 0.05    # Black out 5% of borders to avoid edge noise
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
            if box_area < MIN_AREA_PX: 
                continue            
            # 2. Too Big? (Diagrams/Frames > 1% of screen)
            elif box_area > (img_area * MAX_SCREEN_PCT): 
                continue
            # 3. Aspect Ratio Check (Removes long skinny lines that span > 20% of the screen width/height)
            elif w > (W * 0.2) or h > (H * 0.2):
                continue
            
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

def main():
    # Load config
    if len(os.sys.argv) < 2:
        print("Usage:")
        print(f"\tpython {os.sys.argv[0]} <path_to_config.json>")
        exit(1)
    config = load_config(os.sys.argv[1])

    annotations_path = config['paths']['train_annotations_path']
    
    # 1. Analyze CROHME
    crohme_stats = calculate_crohme_stats(annotations_path)
    
    if True:
        # 2. Analyze Whiteboard (Interactive)
        whiteboard_dir = config['paths']['whiteboard_dir']
        
        # Create directory if it doesn't exist
        if not os.path.exists(whiteboard_dir):
            os.makedirs(whiteboard_dir)
            print(f"Created {whiteboard_dir}. Please put some sample whiteboard images there and run again.")
        else:
            wb_median_area = interactive_whiteboard_calibration(whiteboard_dir)
            #wb_median_area = get_whiteboard_median_via_cc_labeling(whiteboard_dir, debug=True)
            
            if wb_median_area:
                # 3. Calculate Scaling Factor
                # Scaling factor k such that: k^2 * crohme_area = wb_area
                # k = sqrt(wb_area / crohme_area)
                scaling_factor = np.sqrt(wb_median_area / crohme_stats['median_area'])
                
                print(f"\nResults:")
                print(f"Median CROHME Area: {crohme_stats['median_area']:.0f}")
                print(f"Median Whiteboard Area: {wb_median_area:.0f}")
                print(f"Calculated Scaling Factor: {scaling_factor:.2f}")

                # 4. Calculate Anchor Sizes
                # Base anchors on the median size of whiteboard symbols, scaled up and down
                base_size = int(np.sqrt(wb_median_area))
                anchor_sizes = [
                    int(base_size * 0.5),
                    int(base_size * 1.0),
                    int(base_size * 2.0),
                    int(base_size * 4.0)
                ]
                
                update_config(config, {
                    "transform_params": {"scaling_factor": scaling_factor},
                    "model_params": {"anchor_params": {"sizes": anchor_sizes}}
                })
            else:
                print("Skipping calibration update (no whiteboard data collected).")

if __name__ == "__main__":
    main()