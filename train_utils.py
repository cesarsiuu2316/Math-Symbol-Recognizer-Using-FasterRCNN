import os
import numpy as np
import torch
import bisect
from collections import defaultdict, Counter
from torch.utils.data.sampler import BatchSampler
import json
import pandas as pd
from datetime import datetime

class GroupedBatchSampler(BatchSampler):
    """
    Enhanced GroupedBatchSampler.
    Can handle any number of aspect ratio groups automatically.
    """
    def __init__(self, sampler, group_ids, batch_size):
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        self.batch_size = batch_size

    def __iter__(self):
        # defaultdict(list) automatically creates a new list for any new group_id it sees
        buffer_per_group = defaultdict(list)
        
        # Iterate over the base sampler / stream of indices
        for idx in self.sampler:
            # For each index, find its group and add to the corresponding buffer
            group_id = self.group_ids[idx].item()
            buffer_per_group[group_id].append(idx)
            
            # If this specific bucket is full, yield it / flush it
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                buffer_per_group[group_id] = []
        
        # Yield any remaining partial batches from all groups
        for group_id, indices in buffer_per_group.items():
            if len(indices) > 0:
                yield indices

    def __len__(self):
        # Approximate number of batches in the sampler
        return len(self.sampler) // self.batch_size

def create_aspect_ratio_groups(dataset):
    """
    Groups images into groups according to their aspect ratios.
    Args:
        dataset (MathSymbolDataset): The dataset or a subset containing annotations with width and height.
    Returns:
        list: A list of group IDs corresponding to each image in the dataset.
    """
    print("Grouping images by Aspect Ratio (4 Groups)...")
    
    # 1. Handle Subsets (Train/Val Splits)
    # If dataset is a Subset, we need to access the parent dataset via .dataset and map the indices correctly.
    if isinstance(dataset, torch.utils.data.Subset):
        main_dataset = dataset.dataset
        indices = dataset.indices
    else:
        main_dataset = dataset
        indices = range(len(dataset))

    target_min_size = main_dataset.target_min_size
    target_max_size = main_dataset.target_max_size
    scaling_factor = main_dataset.scaling_factor
    
    all_ratios = []
    
    for i in indices:
        # 1. Get Params
        item = main_dataset.annotations[i]
        orig_w = item['width']
        orig_h = item['height']
        
        # 2. Simulate Scaling Transform
        new_w = int(orig_w * scaling_factor)
        new_h = int(orig_h * scaling_factor)

        # 2. Simulate Max Cap
        max_dim = max(new_w, new_h)
        if max_dim > target_max_size:
            scale = target_max_size / max_dim
            new_w = int(new_w * scale)
            new_h = int(new_h * scale)
        
        # 3. Simulate Padding
        final_w = max(new_w, target_min_size)
        final_h = max(new_h, target_min_size)
        
        # 4. Calculate Ratio
        ratio = final_w / final_h
        all_ratios.append(ratio)

    # Dynamic breakpoints based on quantiles
    quantiles = [25, 50, 75]
    all_ratios_np = np.array(all_ratios)
    dynamic_breakpoints = np.percentile(all_ratios_np, quantiles).tolist()

    group_ids = []
    for ratio in all_ratios:
        group_id = bisect.bisect_right(dynamic_breakpoints, ratio)
        group_ids.append(group_id)

    counts = Counter(group_ids)
    print(f"Calculated Breakpoints: {[round(b, 2) for b in dynamic_breakpoints]}")
    print(f"Group Stats: {dict(sorted(counts.items()))}")    
    return group_ids

def save_checkpoint_model(model, optimizer, lr_scheduler, epoch, avg_train_loss, avg_val_loss, train_mAP, val_mAP, chkpt_path_prefix):
    """
    Saves the model checkpoint.
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        lr_scheduler (torch.optim.lr_scheduler): The learning rate scheduler state to save.
        epoch (int): Current epoch number.
        avg_train_loss (float): Average training loss for the epoch.
        avg_val_loss (float): Average validation loss for the epoch.
        train_mAP (float): Training mean Average Precision for the epoch.
        val_mAP (float): Validation mean Average Precision for the epoch.
        chkpt_path_prefix (str): Base path for saving checkpoints.
    Returns:
        None
    """
    # Ensure directory exists
    chkpt_path = f"{chkpt_path_prefix}{epoch}.dat"
    os.makedirs(os.path.dirname(chkpt_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'train_mAP': train_mAP,
        'val_mAP': val_mAP
    }, chkpt_path)
    print(f"Saved checkpoint: {chkpt_path}")

def save_checkpoint_config(config, epoch, chkpt_path_prefix):
    """
    Saves the specific configuration parameters associated with a checkpoint.
    Args: 
        config (dict): The full configuration dictionary.
        epoch (int): Current epoch number.
        chkpt_path_prefix (str): Base path for saving config logs.
    Returns:
        None
    """
    json_path = f"{chkpt_path_prefix}{epoch}.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Extract only the relevant parts of the config
    config_snapshot = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_params": config['model_params'],
        "training_params": config['training_params'],
        "transform_params": config['transform_params']
    }
    
    with open(json_path, 'w') as f:
        json.dump(config_snapshot, f, indent=4)
        
    print(f"Saved config log: {json_path}")

def update_history_log(history_path, epoch, train_loss, val_loss, lr, train_mAP, val_mAP):
    """
    Updates the history CSV log.
    Args:
        history_path (str): Path to the history CSV file.
        epoch (int): Current epoch number.
        train_loss (float): Training loss for the epoch.
        val_loss (float): Validation loss for the epoch.
        lr (float): Learning rate for the epoch.
        train_mAP (float): Training mean Average Precision for the epoch.
        val_mAP (float): Validation mean Average Precision for the epoch.
    Returns:
        None
    """
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    # Create a dictionary for the new row
    new_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Epoch": epoch,
        "Train_Loss": round(train_loss, 5),
        "Val_Loss": round(val_loss, 5),
        "train_mAP": round(train_mAP, 5),
        "val_mAP": round(val_mAP, 5),
        "Learning_Rate": lr,
    }
    
    new_df = pd.DataFrame([new_data])
    
    if os.path.exists(history_path):
        # Append to existing CSV (header=False)
        new_df.to_csv(history_path, mode='a', header=False, index=False)
    else:
        # Create new CSV with header
        new_df.to_csv(history_path, mode='w', header=True, index=False)
        
    print(f"History log updated: {history_path}")

def save_final_report(history_path, report_path, config, total_time_seconds):
    """
    Reads the history CSV and generates a JSON summary report.
    Args:
        history_path (str): Path to the history CSV file.
        report_path (str): Path to save the final report JSON file.
        config (dict): The full configuration dictionary.
        total_time_seconds (float): Total training time in seconds.
    Returns:
        None
    """
    if not os.path.exists(history_path):
        print("Warning: No history log found to generate report.")
        return

    # Load history
    df = pd.read_csv(history_path)
    
    # Calculate Metrics based on MeanAveragePrecision
    best_epoch_mAP_idx = df['val_mAP'].idxmax()
    best_epoch_mAP = int(df.loc[best_epoch_mAP_idx, 'Epoch'])
    best_mAP = float(df.loc[best_epoch_mAP_idx, 'val_mAP'])
    best_epoch_val_loss_idx = df['Val_Loss'].idxmin()
    best_val_loss_epoch = int(df.loc[best_epoch_val_loss_idx, 'Epoch'])
    best_val_loss = float(df.loc[best_epoch_val_loss_idx, 'Val_Loss'])
    
    hours, rem = divmod(total_time_seconds, 3600) 
    minutes, seconds = divmod(rem, 60)
    
    report = {
        "status": "Success",
        "total_duration": "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds),
        "training_summary": {
            "total_epochs": len(df),
            "best_epoch_mAP": best_epoch_mAP,
            "best_val_mAP": best_mAP,
            "best_val_loss_epoch": best_val_loss_epoch,
            "best_val_loss": best_val_loss,
            "final_train_loss": float(df.iloc[-1]['Train_Loss']),
            "final_validation_loss": float(df.iloc[-1]['Val_Loss']),
            "final_train_mAP": float(df.iloc[-1]['train_mAP']),
            "final_val_mAP": float(df.iloc[-1]['val_mAP'])
        },
        "full_configuration": config
    }
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Final report saved: {report_path}")