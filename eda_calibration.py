import os
import numpy as np
from utils import load_config, update_config
import eda_crohme_whiteboard as eda

def main():
    # 1. Load Config
    if len(os.sys.argv) < 2:
        print("Usage:")
        print(f"\tpython {os.sys.argv[0]} <path_to_config.json>")
        exit(1)
    config = load_config(os.sys.argv[1])

    # 2. Get CROHME Stats
    print("\n--- Step 1: CROHME Analysis ---")
    annotations_path = config['paths']['train_annotations_path']
    crohme_stats = eda.calculate_crohme_stats(annotations_path)
    
    if not crohme_stats:
        print("Error: Could not get CROHME stats.")
        return

    # 3. Whiteboard Calibration Setup
    print("\n--- Step 2: Whiteboard Calibration ---")
    whiteboard_dir = config['paths']['whiteboard_dir']
    
    if not os.path.exists(whiteboard_dir):
        os.makedirs(whiteboard_dir)
        print(f"Created {whiteboard_dir}. Please add images and run again.")
        return

    # 4. Check for Existing State and Ask User
    wb_median_area = None
    
    # Check if the json file exists in the current directory
    calibrated_whiteboard_bboxes_path = config['paths']['calibrated_whiteboard_bboxes_path']
    state_exists = os.path.exists(calibrated_whiteboard_bboxes_path)
    
    if state_exists:
        print(f"Found existing calibration data in '{calibrated_whiteboard_bboxes_path}'.")
        print("1. Use saved data (Skip calibration)")
        print("2. Continue calibration (Add new images)")
        print("3. Restart calibration (Delete saved data)")
        
        get_new_anchor_params = config['model_params']['get_new_anchor_params']
        
        if get_new_anchor_params == 1:
            # Load stats without opening GUI
            data = eda.load_state(calibrated_whiteboard_bboxes_path)
            if data:
                all_vals = [val for sublist in data.values() for val in sublist]
                wb_median_area = float(np.median(all_vals))
                print(f"Loaded {len(all_vals)} saved boxes. Median Area: {wb_median_area:.2f}")
            else:
                print("Saved data was empty. Switching to interactive mode.")
                wb_median_area = eda.interactive_whiteboard_calibration(whiteboard_dir, calibrated_whiteboard_bboxes_path, reset=False)
        elif get_new_anchor_params == 2:
            wb_median_area = eda.interactive_whiteboard_calibration(whiteboard_dir, calibrated_whiteboard_bboxes_path, reset=False)
        elif get_new_anchor_params == 3:
            wb_median_area = eda.interactive_whiteboard_calibration(whiteboard_dir, calibrated_whiteboard_bboxes_path, reset=True)
        else:
            print("Invalid get_new_anchor_params. Exiting.")
            return
    else:
        # No state found, start fresh
        print("No saved data found. Starting interactive calibration...")
        wb_median_area = eda.interactive_whiteboard_calibration(whiteboard_dir, calibrated_whiteboard_bboxes_path, reset=False)

    # 5. Process Results & Update Config
    if wb_median_area:
        # Calculate Scaling Factor (Logic from your provided snippet)
        # Scaling factor k such that: k^2 * crohme_area = wb_area
        # k = sqrt(wb_area / crohme_area)
        scaling_factor = np.sqrt(wb_median_area / crohme_stats['median_area'])
        
        print(f"\n--- Final Results ---")
        print(f"Median CROHME Area: {crohme_stats['median_area']:.0f}")
        print(f"Median Whiteboard Area: {wb_median_area:.0f}")
        print(f"Calculated Scaling Factor: {scaling_factor:.2f}")

        # Calculate Anchor Sizes
        base_size = int(np.sqrt(wb_median_area))
        anchor_sizes = [
            int(base_size * 0.5),
            int(base_size * 1.0),
            int(base_size * 2.0),
            int(base_size * 4.0)
        ]        
        print(f"New Anchor Sizes: {anchor_sizes}")

        # Update Config
        update_config(config, {
            "transform_params": {"scaling_factor": scaling_factor},
            "model_params": {"anchor_params": {"sizes": anchor_sizes}}
        })
        print(f"Configuration file '{config["paths"]["config_path"]}' updated successfully.")
    else:
        print("Skipping calibration update (no whiteboard data collected).")

if __name__ == "__main__":
    main()