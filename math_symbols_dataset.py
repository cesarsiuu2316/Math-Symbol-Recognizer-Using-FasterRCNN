import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import json
import random
from utils import load_config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A

class MathSymbolDataset(Dataset):
    """
    PyTorch Dataset for Math Symbol Detection.
    Handles domain shift problem through adaptation of the images via resizing and morphological + noise augmentation.
    """
    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config
        self.img_dir = config['paths']['synthetic_image_dir']

        # Load Synthetic annotations
        synthetic_annotations_path = config['paths']['synthetic_annotations_path']
        with open(synthetic_annotations_path, 'r') as f:
            self.annotations = json.load(f)

        # Only keep the list of annotations
        self.annotations = self.annotations['annotations']
            
        # Load class mapping to ensure correct IDs
        class_mapping_path = config['paths']['class_mapping_path']
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        # Transformation parameters
        self.transform = config['transform_params']['transform']

        # Augmentation parameters
        augmentation_params = config['transform_params']['augmentation_params']
        self.blur_kernels = augmentation_params['blur_kernels']
        self.noise_sigma_range = augmentation_params['noise_sigma_range']
        self.threshold_factor_range = augmentation_params['threshold_factor_range']

    def __len__(self):
        return len(self.annotations)

    def __noise_augmentation(self, img):
        """
        Applies 'analog' noise to digital ink to simulate whiteboard markers.
        Args: 
            img (numpy.ndarray): Grayscale image (H, W) with white background and black ink.
        Returns:
            numpy.ndarray: Image with blur, noise and thresholding applied.
        """

        if self.transform:
            # 2. Blur + Noise + Threshold
            if random.random() < 0.8:
                # A. Blur to create gray transition areas
                blur_amount = random.choice(self.blur_kernels)
                img_blurred = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
                #print(f"Blur Amount: {blur_amount}")
                
                # B. Add Gaussian Noise            
                noise_sigma = random.randint(self.noise_sigma_range[0], self.noise_sigma_range[1])
                noise = np.random.normal(0, noise_sigma, img_blurred.shape).astype(np.int16)
                img_noisy = img_blurred.astype(np.int16) + noise
                img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
                img = img_noisy

        # C. Adaptive thresholding based on dynamic range
        min_val = np.min(img) # Darkest pixel (The Ink)
        max_val = np.max(img) # Brightest pixel (The Background)
        dynamic_range = max_val - min_val
            
        if dynamic_range < 20:
            return img
                
        # The threshold value is set to a random factor of the dynamic range above the min_val
        thresh_factor = random.uniform(self.threshold_factor_range[0], self.threshold_factor_range[1]) 
        threshold_val = (dynamic_range * thresh_factor) + min_val
        
        _, img_result = cv2.threshold(img, int(threshold_val), 255, cv2.THRESH_BINARY)
        # print(f"Noise Sigma: {noise_sigma}, Threshold Value: {threshold_val:.2f}, Thresh Factor: {thresh_factor:.2f}")
        return img_result

    def __getitem__(self, idx):
        # 1. Load Data
        item = self.annotations[idx]
        img_path = os.path.join(self.img_dir, item['image_name'])
        
        # Load image (BGR)
        img = cv2.imread(img_path) # loads image in BGR format and HWC (height, width, channels) shape
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Convert to Grayscale for augmentation, then back to RGB for model
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        boxes = np.array(item['boxes'], dtype=np.float32)
        labels = torch.tensor([self.class_mapping[l] for l in item['labels']], dtype=torch.int64)

        # Apply Global Noise + Blur + Threshold Augmentation
        img_gray = self.__noise_augmentation(img_gray)

        # Convert to RGB for model input (faster rcnn expects 3 channels)
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        # 5. Convert to Tensor for model input
        # Normalize to [0, 1] and permute to [C, H, W] (Channel, Height, Width)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 6. Prepare Target Dict
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        # Area
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["area"] = torch.as_tensor(area, dtype=torch.float32)
        else:
            target["area"] = torch.as_tensor([], dtype=torch.float32)
            print(f"Warning: No boxes for image {img_path}")
            
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64)

        return img_tensor, target

def collate_fn(batch):
    """
    Custom collate function for variable size images.
    Returns tuple of (images, targets).
    """
    return tuple(zip(*batch))

def test_dataset(dataset, image_id=0):
    # Test the dataset    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    img, target = dataset[image_id]
    print(f"Image shape: {img.shape}")
    print(f"Target keys: {target.keys()}")
    print(f"Number of boxes: {len(target['boxes'])}")
    
    # Visualize
    # Convert back to numpy [H, W, C]
    original_images_path = dataset.img_dir
    original_img_path = os.path.join(original_images_path, os.path.basename(dataset.annotations[image_id]['image_name']))
    original_img = cv2.imread(original_img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_np = img.permute(1, 2, 0).numpy()
    
    # Plot original image with boxes and augmentated image
    _, ax = plt.subplots(1, 2, figsize=(8, 16))
    ax[0].imshow(original_img)
    ax[0].set_title("Original Image with Boxes")
    for box in dataset.annotations[image_id]['boxes']:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
    
    ax[1].imshow(img_np)
    ax[1].set_title("Augmented Image with Boxes")
    for box in target['boxes']:
        x1, y1, x2, y2 = box.numpy()
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax[1].add_patch(rect)
        
    plt.show()

def main():
    # Load config
    if len(os.sys.argv) < 2:
        print("Usage:")
        print(f"\tpython {os.sys.argv[0]} <path_to_config.json>")
        exit(1)
    config = load_config(os.sys.argv[1])

    dataset = MathSymbolDataset(config)

    for _ in range(0, 10):
        image_id = random.randint(0, 10000)
        test_dataset(dataset, image_id)

if __name__ == "__main__":
    main()
