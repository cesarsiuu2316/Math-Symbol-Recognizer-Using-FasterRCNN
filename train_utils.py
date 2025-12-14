import os
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
    # If dataset is a Subset, we need to access the parent dataset via .dataset
    # and map the indices correctly.
    if isinstance(dataset, torch.utils.data.Subset):
        main_dataset = dataset.dataset
        indices = dataset.indices
    else:
        main_dataset = dataset
        indices = range(len(dataset))

    target_min_size = main_dataset.target_min_size
    scaling_factor = main_dataset.scaling_factor
    
    # --- CONFIGURATION: Aspect ratio breakpoints ---
    # Everything < 1 goes to Group 0
    # Everything 1 to 1.5 goes to Group 1
    # Everything 1.5 to 2 goes to Group 2
    # Everything >= 2 goes to Group 3
    breakpoints = [1.1, 1.5, 2] 
    group_ids = []
    
    for i in indices:
        # 1. Get Params
        item = main_dataset.annotations[i]
        orig_w = item['width']
        orig_h = item['height']
        
        # 2. Simulate Transform
        new_w = int(orig_w * scaling_factor)
        new_h = int(orig_h * scaling_factor)
        
        # 3. Simulate Padding
        final_w = max(new_w, target_min_size)
        final_h = max(new_h, target_min_size)
        
        # 4. Calculate Ratio
        ratio = final_w / final_h
        
        # 5. Assign Group using bisect
        group_id = bisect.bisect_right(breakpoints, ratio)
        group_ids.append(group_id)

    counts = Counter(group_ids)
    print(f"Group Stats: {dict(sorted(counts.items()))}")
    
    return group_ids

def save_checkpoint_model(model, optimizer, lr_scheduler, epoch, avg_train_loss, avg_val_loss, chkpt_path_prefix):
    """
    Saves the model checkpoint.
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        lr_scheduler (torch.optim.lr_scheduler): The learning rate scheduler state to save.
        epoch (int): Current epoch number.
        avg_train_loss (float): Average training loss for the epoch.
        avg_val_loss (float): Average validation loss for the epoch.
        chkpt_path_config (str): Base path for saving checkpoints.
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
        'val_loss': avg_val_loss
    }, chkpt_path)
    print(f"Saved checkpoint: {chkpt_path}")

def save_checkpoint_config(config, epoch, chkpt_path_prefix):
    """
    Saves the specific configuration parameters associated with a checkpoint.
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

def update_history_log(history_path, epoch, train_loss, val_loss, lr):
    """
    Updates the history CSV log.
    """
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    # Create a dictionary for the new row
    new_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Epoch": epoch,
        "Train_Loss": round(train_loss, 5),
        "Val_Loss": round(val_loss, 5),
        "Learning_Rate": lr
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
    """
    if not os.path.exists(history_path):
        print("Warning: No history log found to generate report.")
        return

    # Load history
    df = pd.read_csv(history_path)
    
    # Calculate Metrics
    best_epoch_idx = df['Val_Loss'].idxmin()
    best_epoch = int(df.loc[best_epoch_idx, 'Epoch'])
    best_val_loss = float(df.loc[best_epoch_idx, 'Val_Loss'])
    
    hours, rem = divmod(total_time_seconds, 3600) 
    minutes, seconds = divmod(rem, 60)
    
    report = {
        "status": "Success",
        "total_duration": "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds),
        "training_summary": {
            "total_epochs": len(df),
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "final_train_loss": float(df.iloc[-1]['Train_Loss']),
            "final_validation_loss": float(df.iloc[-1]['Val_Loss'])
        },
        "full_configuration": config
    }
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Final report saved: {report_path}")