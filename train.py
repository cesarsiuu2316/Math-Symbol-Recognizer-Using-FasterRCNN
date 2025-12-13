import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import os
import time
from utils import load_config
from model import get_model
from math_symbols_dataset import MathSymbolDataset, collate_fn
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    running_loss = 0.0
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)

        # Send all target tensors to device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets) # Returns a dict of losses
        losses = sum(loss for loss in loss_dict.values()) # Total loss

        optimizer.zero_grad() # Zero the gradients
        losses.backward() # Backpropagate
        optimizer.step() # Update weights

        loss_value = losses.item() # Get scalar loss value
        running_loss += loss_value # Accumulate loss

        if (i + 1) % print_freq == 0:
            print(f"Epoch [{epoch}] Batch [{i+1}/{len(data_loader)}] Loss: {loss_value:.4f}")

    return running_loss / len(data_loader)

def main():
    if len(os.sys.argv) < 2:
        print("Usage: python train.py <config.json>")
        os.sys.exit(1)

    config = load_config(os.sys.argv[1])
    
    # 1. Setup Device
    device = torch.device(config['training_params']['device'])
    print(f"Training on: {device}")

    # 2. Create Custom Dataset Instance
    dataset = MathSymbolDataset(config)
    
    # 3. Setup Grouped Batch Sampler
    # Create random sampler (indexes)
    train_sampler = RandomSampler(dataset)
    # Calculate groups (0 or 1) using Annotations json data
    group_ids = create_aspect_ratio_groups(dataset)
    # Get batch size from config
    batch_size = config['training_params']['batch_size']
    # Create Custom GroupedBatchSampler Instance
    batch_sampler = GroupedBatchSampler(
        train_sampler, 
        group_ids, 
        batch_size
    )

    # 4. DataLoader
    num_workers = config['training_params']['num_workers']
    data_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler, 
        num_workers=num_workers,
        collate_fn=collate_fn
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

    # 6. Training Loop
    num_epochs = config['training_params']['num_epochs']
    output_dir = config['paths']['output_dir']
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Start Training ---")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print_freq = config['training_params']['print_freq']
        avg_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq)
        lr_scheduler.step()
        print(f"Epoch {epoch} Done. Average Loss: {avg_loss:.4f}")
        
        # Save
        chkpt_path_config = config['paths']['model_checkpoint_prefix']
        chkpt_path = f"{chkpt_path_config}epoch_{epoch}.dat"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, chkpt_path)

    print(f"Training finished in {time.time() - start_time:.0f} seconds.")

if __name__ == "__main__":
    main()