import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
import os
import time
from utils import load_config
from model import get_model
from math_symbols_dataset import MathSymbolDataset, collate_fn
from train_utils import (GroupedBatchSampler, create_aspect_ratio_groups, save_checkpoint_model, 
                        save_checkpoint_config, update_history_log, save_final_report)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
import cv2

def visualize_debug_batch(images, targets, batch_size, predictions, epoch, id_to_name, debug_output_dir, score_threshold=0.5):
    """
    Saves the first image of the batch with GT (Green) and Preds (Red) boxes.
    Args:
        images: List of image tensors in the batch.
        targets: List of target dicts in the batch.
        predictions: List of prediction dicts from the model.
        epoch (int): Current epoch number.
        debug_output_dir (str): Directory to save debug images.
        score_threshold (float): Score threshold for displaying predictions (Set to 0 to show all weak boxes).
    Returns: 
        None
    """
    # 1. Get the first 5 images in the batch
    # Image is Tensor [C, H, W] -> Move to CPU -> Numpy -> Transpose to [H, W, C]
    max_images = min(5, batch_size) # Limit to first 5 images
    for i in range(max_images):
        img_tensor = images[i].cpu()
        img_np = img_tensor.permute(1, 2, 0).numpy()
        # Scale from [0, 1] to [0, 255] and convert to contiguous array for OpenCV
        img_np = (img_np * 255).astype(np.uint8)
        img_np = np.ascontiguousarray(img_np)
        # Convert RGB to BGR for OpenCV
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 2. Draw Ground Truth (Green)
        gt_boxes = targets[i]['boxes'].cpu().numpy()
        for box in gt_boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 3. Draw Predictions (Red)
        # Filter by score (only show confident predictions)
        pred_boxes = predictions[i]['boxes'].cpu().numpy()
        pred_scores = predictions[i]['scores'].cpu().numpy()
        pred_labels = predictions[i]['labels'].cpu().numpy()
        
        for box, score, label_id in zip(pred_boxes, pred_scores, pred_labels):
            if score > score_threshold:
                x1, y1, x2, y2 = box.astype(int)
                class_name = id_to_name.get(int(label_id), str(label_id))
                label_text = f"{class_name}"
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(img_np, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 255), 1)

        # 4. Save
        save_path = os.path.join(debug_output_dir, f"epoch_{epoch}_img_{i}.png")
        cv2.imwrite(save_path, img_np)
    print(f"Saved debug image to {save_path}")

def train_one_epoch(model, optimizer, data_loader, device, epoch, max_norm, print_freq, id_to_name, debug=False, debug_output_dir=""):
    """
    Trains the model for one epoch.
    Args:
        model: The model to train.
        optimizer: The optimizer for training.
        data_loader: DataLoader for training data.
        device: Device to run the model on.
        epoch (int): Current epoch number.
        max_norm (float): Maximum norm for gradient clipping.
        print_freq (int): Frequency of printing training status.
    Returns:
        Average Training Loss (float): Average loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    
    for i, (images, targets) in enumerate(data_loader):
        optimizer.zero_grad() # Zero gradients
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Send all images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Debug Visualization for first batch
        if debug and i == 0: 
            model.eval()
            with torch.no_grad():
                predictions = model(images)
            batch_size = len(images)
            visualize_debug_batch(images, targets, batch_size, predictions, epoch, id_to_name, debug_output_dir, 0.5)
            model.train()

        loss_dict = model(images, targets) # Returns a dict of losses
        losses = sum(loss for loss in loss_dict.values()) # Total loss

        losses.backward() # Backpropagate

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step() # Update weights

        loss_value = losses.item() # Get scalar loss value
        running_loss += loss_value # Accumulate loss

        if (i + 1) % print_freq == 0:
            print(f"Epoch [{epoch}] Batch [{i+1}/{len(data_loader)}] Loss: {loss_value:.4f}")

    avg_loss = running_loss / len(data_loader)
    return avg_loss

@torch.no_grad()
def validate_loss_one_epoch(model, data_loader, device, epoch, id_to_name, debug=False, debug_output_dir=""):
    """
    Calculates validation loss in train mode to get loss dict. 
    But with torch.no_grad() to avoid computing gradients.
    But since the model's 
    Args:
        model: The model to validate.
        data_loader: DataLoader for validation data.
        device: Device to run the model on.
        epoch (int): Current epoch number.
        debug (bool): If True, enables debug visualization for the first batch.
        debug_output_dir (str): Directory to save debug images if debug is True.
    Returns:
        Validation Loss (float): Average validation loss.
    """
    model.train() # Keeping in train mode to get loss dict
    running_loss = 0.0

    for i, (images, targets) in enumerate(data_loader):
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        # Send images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Debug Visualization for first batch
        if debug and i == 0: 
            model.eval()
            predictions = model(images)
            batch_size = len(images)
            visualize_debug_batch(images, targets, batch_size, predictions, epoch, id_to_name, debug_output_dir, 0.5)
            model.train()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()

    avg_loss = running_loss / len(data_loader)
    return avg_loss

@torch.no_grad()
def evaluate_map(model, data_loader, device):
    """
    Calculates Mean Average Precision (mAP).
    Requires model in .eval() mode.
    Args:
        model: The model to evaluate.
        data_loader: DataLoader for evaluation data.
        device: Device to run the model on.
    Returns:
        mAP (float): Mean Average Precision over IoU thresholds from 0.5 to 0.95)
    """
    model.eval()
    metric = MeanAveragePrecision()
    
    for images, targets in data_loader:
        # 1. Move images to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # 3. Get Predictions
        preds = model(images)
        # 4. Update Metric 
        metric.update(preds, targets)

    # 5. Compute Final mAP (Returns a dict with 'map', 'map_50', 'map_75', etc.)
    result = metric.compute()
    
    # mAP is the mean average precision over IoU thresholds from 0.5 to 0.95
    map_score = result['map'].item()
    return map_score

def test_data_loader(data_loader, device):
    """
    Tests the DataLoader by iterating through one batch and printing shapes.
    Args:
        data_loader: DataLoader to test.
        device: Device to move data to.
    Returns:
        None
    """ 
    print("Testing DataLoader...50 batches")
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        print(f"Batch Size: {len(images)}")
        for j in range(len(images)):
            print(f" Image {j} shape: {images[j].shape}")
            print(f" Target {j} boxes shape: {targets[j]['boxes'].shape}")
        if i == 50:
            cv2.imshow("Test Image", images[0].permute(1, 2, 0).cpu().numpy())
            cv2.waitKey(0)
            break

def main():
    checkpoint_path = None
    if len(os.sys.argv) < 2:
        print(f"Usage: python {os.sys.argv[0]} <config.json>")
        print(f"\tOptionally: python {os.sys.argv[0]} <config.json> <checkpoint_path>")
        os.sys.exit(1)
    elif len(os.sys.argv) == 3:
        print("Loading checkpoint...")
        checkpoint_path = os.sys.argv[2]
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file not found at {checkpoint_path}")
            os.sys.exit(1)

    config = load_config(os.sys.argv[1])
    if (checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")

    # 1. Load Class Mapping (Invert it: {'alpha': 1} -> {1: 'alpha'})
    mapping_path = config['paths']['class_mapping_path']
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    id_to_name = {v: k for k, v in class_mapping.items()}
    
    # 1. Setup Device
    device = torch.device(config['training_params']['device'])
    print(f"Training on: {device}")

    # 2. Create Custom Dataset Instance
    dataset = MathSymbolDataset(config)

    random_generator_seed = config['training_params']['random_generator_seed']
    random_generator = torch.Generator().manual_seed(random_generator_seed) # For reproducibility
    train_split_ratio = config['training_params']['train_split_ratio']
    train_size = int(train_split_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=random_generator)
    
    print(f"Dataset Split: {len(train_dataset)} Train, {len(val_dataset)} Validation")

    # 3. Setup Grouped Batch Sampler
    # Create random sampler (indexes)
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    # Calculate groups based on aspect ratios
    train_group_ids = create_aspect_ratio_groups(train_dataset)
    val_group_ids = create_aspect_ratio_groups(val_dataset)
    # Get batch size from config
    batch_size = config['training_params']['batch_size']
    # Create Custom GroupedBatchSampler Instance
    train_batch_sampler = GroupedBatchSampler(train_sampler, train_group_ids, batch_size)
    val_batch_sampler = GroupedBatchSampler(val_sampler, val_group_ids, batch_size)

    # 4. DataLoader
    num_workers = config['training_params']['num_workers']
    pin_memory = True if device.type == 'cuda' else False

    train_data_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )

    # 5. Model & Optimizer
    model = get_model(config)
    model.to(device)
    
    learning_rate = config['training_params']['learning_rate']
    weight_decay = config['training_params']['weight_decay']
    # Filter only trainable parameters for optimizer
    params = [p for p in model.parameters() if p.requires_grad] # Filter only trainable parameters
    optimizer = optim.AdamW(
        params, 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    step_size = config['training_params']['lr_scheduler']['step_size']
    gamma = config['training_params']['lr_scheduler']['gamma']
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )

    # --- RESUME FROM CHECKPOINT ---
    start_epoch = 1
    if checkpoint_path:
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # Set start epoch
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # 6. Training Loop
    num_epochs = config['training_params']['num_epochs']
    output_dir = config['paths']['output_dir']
    max_norm = config['training_params']['grad_clip_max_norm']
    print_freq = config['training_params']['print_freq']
    debug_output_dir = config['paths']['debug_output_dir']
    debug = config['training_params']['debug']
    chkpt_path_prefix = config['paths']['model_checkpoint_prefix']
    history_path = config['paths']['history_log_path']
    report_path = config['paths']['final_report_path']
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_output_dir, exist_ok=True)
    
    print("\n--- Start Training ---")
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs + 1):
        # Train One Epoch
        avg_train_loss = train_one_epoch(
            model=model, 
            optimizer=optimizer, 
            data_loader=train_data_loader, 
            device=device, 
            epoch=epoch, 
            max_norm=max_norm, 
            print_freq=print_freq,
            id_to_name=id_to_name,
            debug=debug,
            debug_output_dir=debug_output_dir
        )
        # Validate One Epoch
        #avg_val_loss = 0.0
        avg_val_loss = validate_loss_one_epoch(
            model=model, 
            data_loader=val_data_loader, 
            device=device, 
            epoch=epoch,
            id_to_name=id_to_name,
            debug=False,
            debug_output_dir=""
        )
        # Compute mAP
        mAP_train_score = 0.0
        #mAP_train_score = evaluate_map(model, train_data_loader, device)
        mAP_val_score = evaluate_map(model, val_data_loader, device)
        # Scheduler Step
        lr_scheduler.step()
        print(f"Epoch {epoch} Done. \tAvg Training Loss: {avg_train_loss:.4f} \tAvg Validation Loss: {avg_val_loss:.4f} \tmAP Val: {mAP_val_score:.4f}")
        #print(f"Epoch {epoch} Done. \tAvg Training Loss: {avg_train_loss:.4f} \tmAP Train: {mAP_train_score:.4f} \tmAP Val: {mAP_val_score:.4f}")
        # --- LOGGING AND SAVING ---
        # Save Checkpoint
        save_checkpoint_model(
            model=model, 
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            epoch=epoch, 
            avg_train_loss=avg_train_loss, 
            avg_val_loss=avg_val_loss, 
            train_mAP=mAP_train_score,
            val_mAP=mAP_val_score, 
            chkpt_path_prefix=chkpt_path_prefix
        )
        # Save Config
        save_checkpoint_config(config=config, epoch=epoch, chkpt_path_prefix=chkpt_path_prefix)
        # Update History Log
        current_lr = optimizer.param_groups[0]['lr']
        update_history_log(
            history_path=history_path, 
            epoch=epoch, 
            train_loss=avg_train_loss, 
            val_loss=avg_val_loss, 
            lr=current_lr, 
            train_mAP=mAP_train_score,
            val_mAP=mAP_val_score
        )

    # End of Training
    total_time = time.time() - start_time

    # Save Final Report (.json)
    save_final_report(
        history_path=history_path, 
        report_path=report_path, 
        config=config, 
        total_time_seconds=total_time
    )

    # Save Final Model
    final_model_path = config['paths']['final_model_path']
    torch.save(model.state_dict(), final_model_path)

    print(f"Training finished in {total_time:.0f} seconds.")

if __name__ == "__main__":
    main()