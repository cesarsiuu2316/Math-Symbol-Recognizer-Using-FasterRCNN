import torch
import cv2
import os
import json
import concurrent.futures
import pickle

from utils import load_config
from model import get_model
from torch.utils.data import DataLoader
from inference_dataset import InferenceDataset, collate_fn
from inference_utils import create_exact_dimension_groups, GroupedBatchSampler
from torch.utils.data import SequentialSampler


def load_trained_model(config, model_path=None):
    """
    Loads the config and the trained model weights.
    """
    device = torch.device(config['inference_params']['device'])

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


def get_all_image_paths(input_directory):
    """
    Recursively finds all images. 
    """
    image_paths = []
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def save_img_with_predictions(boxes, labels, scores, threshold, id_to_name, image_path, output_path):
    """
    Save an image with bounding boxes drawn around detected symbols.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            x1, y1, x2, y2 = box.astype(int)
            class_name = id_to_name.get(label, f"Unknown ({label})")
            label = f"{class_name}: {int(score * 100)}%"
            
            # Draw Box (Red)
            # Color varies according to score percentage
            # Greener > 90 to Yellow > 80 to Redder > 70
            if score >= 0.9:
                color = (0, 255, 0) # Green
            elif score >= 0.8:
                color = (0, 255, 255) # Yellow
            else:
                color = (0, 0, 255) # Red
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            
            # Draw Label (White text with Red background)
            cv2.putText(img, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    cv2.imwrite(output_path, img)


def inference_on_folder(input_dir, output_dir, all_files, model, device, id_to_name, detection_threshold, batch_size, pin_memory, batch_workers, verbose=False):
    if verbose:
        print("... scanning input files ... ", flush=True)
    total_files = len(all_files)

    if verbose:
        print(f"Found {total_files} images. Starting Inference...", flush=True)

    dataset = InferenceDataset(all_files)

    group_ids = create_exact_dimension_groups(all_files, verbose=verbose)
    sampler = SequentialSampler(dataset)
    batch_sampler = GroupedBatchSampler(sampler, group_ids, batch_size)

    data_loader = None

    data_loader = DataLoader(
        dataset, 
        batch_sampler=batch_sampler,
        num_workers=batch_workers, 
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    all_data = []
    total_symbols = 0
    total_files_proc = 0
    created_dirs = set()
    
    # Thread pool for saving images in the background
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=batch_workers)
    
    with torch.no_grad():
        for batch_imgs, batch_paths in data_loader:
            try:
                # Get dimensions of the first image
                h_tens, w_tens = batch_imgs[0].shape[-2:]
                # 1. Set min_size to the smallest dimension of the image
                model.transform.min_size = (int(min(h_tens, w_tens)),)
                # 2. Set max_size to the largest dimension of the image
                model.transform.max_size = int(max(h_tens, w_tens))
            except Exception as e:
                print(f"Warning: Could not update model transform size: {e}")

            images = list(img.to(device) for img in batch_imgs) # Images to device          
            predictions = model(images) # inference

            for path, pred in zip(batch_paths, predictions):
                # Clean up the path string if needed (relative path)
                rel_filename = os.path.relpath(path, input_dir)
                name_image = os.path.basename(path)
                
                # Extract data from GPU tensor to CPU list
                boxes = pred['boxes'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()

                if output_dir is not None:
                    # save image with prediction in new folder with same image name
                    new_output_dir = os.path.join(output_dir, os.path.dirname(rel_filename))
                    output_path = os.path.join(new_output_dir, name_image)

                    if new_output_dir not in created_dirs:
                        os.makedirs(new_output_dir, exist_ok=True)
                        created_dirs.add(new_output_dir)

                    executor.submit(
                        save_img_with_predictions,
                        boxes, labels, scores, detection_threshold, id_to_name, path, output_path
                    )
                
                # Convert to desired format
                file_data = []
                for box, label, score in zip(boxes, labels, scores):
                    if score >= detection_threshold:
                        x, y, x2, y2 = box
                        w = x2 - x
                        h = y2 - y
                        class_name = id_to_name.get(label, f"Unknown ({label})")
                        file_data.append((x, y, w, h, {class_name: score}))
                
                total_symbols += len(file_data)
                all_data.append((rel_filename, file_data))
                total_files_proc += 1

                if verbose and total_files_proc % 5 == 0:
                    print(f"Processed {total_files_proc}/{total_files} files...", end='\r', flush=True)

        if verbose:
            print("\nWaiting for remaining images to finish saving to disk...", flush=True)
        executor.shutdown(wait=True)

        if verbose:
            print("A total of {0:d} symbols from {1:d} files were extracted".format(total_symbols, len(all_data)))

    return all_data


def main():
    # 1. Load Config and Model
    model_path = None
    inference_folder_path = None
    config_path = None

    if len(os.sys.argv) < 3:
        print(f"Usage: python {os.sys.argv[0]} <config.json> <inference_folder_path>")
        print(f"\tOptionally: python {os.sys.argv[0]} <config.json> <inference_folder_path> <checkpoint_path>")
        os.sys.exit(1)

    if len(os.sys.argv) == 3:
        config_path = os.sys.argv[1]
        inference_folder_path = os.sys.argv[2]
        
    elif len(os.sys.argv) == 4:
        config_path = os.sys.argv[1]
        inference_folder_path = os.sys.argv[2]
        model_path = os.sys.argv[3]
            
    # 1. Load config and model
    try:
        config = load_config(config_path)
        model, device = load_trained_model(config, model_path)
        id_to_name = get_class_names(config)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 2. gather image paths
    image_paths = get_all_image_paths(inference_folder_path)
    if len(image_paths) == 0:
        print(f"No images found in {inference_folder_path}")
        return
    
    # 3. Output directory setup: results/<folder_name>
    folder_name = os.path.basename(os.path.normpath(inference_folder_path))
    output_dir = os.path.join("results", folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # 4. Inference params:
    config_inference = config['inference_params']
    detection_threshold = config_inference['detection_threshold']
    batch_size = config_inference['batch_size']
    pin_memory = config_inference['pin_memory']
    batch_workers = config_inference['num_workers']
    
    # 5. Run Inference on folder
    all_data = inference_on_folder(
        input_dir=inference_folder_path,
        output_dir=output_dir,
        all_files=image_paths,
        model=model,
        device=device,
        id_to_name=id_to_name,
        detection_threshold=detection_threshold,
        batch_size=batch_size,
        pin_memory=pin_memory,
        batch_workers=batch_workers,
        verbose=True
    )

    # 6. Save data
    output_filename = os.path.join(output_dir, f"{folder_name}_predictions.pkl")
    with open(output_filename, "wb") as output_file:
        pickle.dump(all_data, output_file, pickle.HIGHEST_PROTOCOL)

    # 7. Save configs for reproducibility
    config_output_path = os.path.join(output_dir, "config_used.json")
    with open(config_output_path, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    main()