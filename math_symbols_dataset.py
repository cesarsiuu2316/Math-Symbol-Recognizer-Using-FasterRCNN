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
        self.img_dir = config['paths']['train_img_dir']

        # Load annotations
        annotations_path = config['paths']['train_annotations_path']
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)

        # Only keep the list of annotations
        self.annotations = self.annotations['annotations']
            
        # Load class mapping to ensure correct IDs
        class_mapping_path = config['paths']['class_mapping_path']
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        # Transformation parameters
        self.transform = config['transform_params']['transform']
        self.scaling_factor = config['transform_params']['scaling_factor']
        self.target_min_size = config['transform_params']['target_min_size']
        # Augmentation parameters
        augmentation_params = config['transform_params']['augmentation_params']
        self.morphological_kernels = augmentation_params['morphological_kernels']
        self.blur_kernels = augmentation_params['blur_kernels']
        self.noise_sigma_range = augmentation_params['noise_sigma_range']
        self.threshold_range = augmentation_params['threshold_range']
        

    def __len__(self):
        return len(self.annotations)

    def __mimic_whiteboard_ink(self, img):
        """
        Applies 'analog' noise to digital ink to simulate whiteboard markers.
        Input: Binary Image (0=Ink, 255=Background)
        Output: Noisy Binary Image
        """
        # 1. Random Global Thickness (Marker Tip Size)
        if random.random() < 0.5:
            morpho_kernel_size = random.choice(self.morphological_kernels)
            kernel = np.ones((morpho_kernel_size, morpho_kernel_size), np.uint8)
            op = random.choice(['erode', 'dilate', 'none'])
            if op == 'erode':
                # Erode image = Min filter = Expands Black (Ink gets Thicker)
                img = cv2.erode(img, kernel, iterations=1)
            elif op == 'dilate':
                # Dilate image = Max filter = Expands White (Ink gets Thinner)
                img = cv2.dilate(img, kernel, iterations=1)

        # 2. "Expert" Trick usually used: Blur + Noise + Threshold
        # This creates a "rough edge" and "ink skip" look
        # Keep 15% clean to prevent over-noising
        if random.random() < 0.85:
            # A. Blur to create gray transition areas
            blur_amount = random.choice(self.blur_kernels)
            img_blurred = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
            
            # B. Add Gaussian Noise
            # This pushes some gray pixels to black (blobs) and some to white (holes)
            noise_sigma = random.randint(self.noise_sigma_range[0], self.noise_sigma_range[1]) # How "grainy" is the camera?
            noise = np.random.normal(0, noise_sigma, img_blurred.shape).astype(np.int16)
            
            # Add noise to image (convert to int16 to avoid overflow, then clip)
            img_noisy = img_blurred.astype(np.int16) + noise
            img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
            
            # C. Re-Binarize (Simulate the whiteboard algorithm)
            # The threshold value determines if the result is heavy or light ink
            threshold_val = random.randint(self.threshold_range[0], self.threshold_range[1])
            _, img_result = cv2.threshold(img_noisy, threshold_val, 255, cv2.THRESH_BINARY)
    
            return img_result
        
        return img

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

        # 2. Dynamic Scaling
        # Resize image and boxes to match whiteboard domain size
        h, w = img_gray.shape
        new_w = int(w * self.scaling_factor)
        new_h = int(h * self.scaling_factor)
        
        img_gray = cv2.resize(img_gray, (new_w, new_h))
        
        # Scale boxes: [x1, y1, x2, y2]
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= (new_w / w)
            boxes[:, [1, 3]] *= (new_h / h)

        # 3. Padding
        # Pad with white pixels if smaller than target_min_size
        # Faster R-CNN needs a minimum size to generate feature maps
        pad_total_w = max(0, self.target_min_size - new_w) # if negative, pad_w = 0, no padding needed / else pad_w = amount to pad
        pad_total_h = max(0, self.target_min_size - new_h)
        
        if pad_total_w > 0 or pad_total_h > 0:
            # Split padding to center the image
            pad_top = pad_total_h // 2
            pad_bottom = pad_total_h - pad_top
            pad_left = pad_total_w // 2
            pad_right = pad_total_w - pad_left

            # Apply Padding
            img_gray = cv2.copyMakeBorder(
                img_gray, 
                top=pad_top, bottom=pad_bottom, 
                left=pad_left, right=pad_right, 
                borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            
            # Shift boxes to match the new image position
            if len(boxes) > 0:
                boxes[:, 0] += pad_left
                boxes[:, 2] += pad_left
                boxes[:, 1] += pad_top
                boxes[:, 3] += pad_top

        # 4. Domain Augmentation (Morphological)
        # Change ink thickness sometimes + blur + noise + threshold
        if self.transform:
            img = self.__mimic_whiteboard_ink(img_gray)

        # Convert to RGB for model input (faster rcnn expects 3 channels)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

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
    config = load_config()
    dataset = MathSymbolDataset(config)
    for _ in range(0, 5):
        image_id = random.randint(0, 10000)
        test_dataset(dataset, image_id)

if __name__ == "__main__":
    main()
