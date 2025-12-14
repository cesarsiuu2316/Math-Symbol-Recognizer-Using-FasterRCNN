import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
import os
import time
from utils import load_config
from model import get_model
from math_symbols_dataset import MathSymbolDataset, collate_fn
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups

def train_one_epoch(model, optimizer, data_loader, device, epoch, max_norm, print_freq):
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
        float: Average loss over the epoch.
    """
    model.train()
    running_loss = 0.0
    
    for i, (images, targets) in enumerate(data_loader):
        # Send all images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets) # Returns a dict of losses
        losses = sum(loss for loss in loss_dict.values()) # Total loss

        optimizer.zero_grad() # Zero gradients
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
def validate_loss_one_epoch(model, data_loader, device, debug=False):
    """
    Calculates validation loss in train mode to get loss dict. 
    But with torch.no_grad() to avoid computing gradients.
    Args:
        model: The model to validate.
        data_loader: DataLoader for validation data.
        device: Device to run the model on.
        epoch (int): Current epoch number.
        debug (bool): If True, prints additional debug information.
    Returns:
        float: Average validation loss.
    """
    model.train() # Keeping in train mode to get loss dict
    running_loss = 0.0
    
    print("Validating...")
    for images, targets in data_loader:
        # Send images and targets to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()

    avg_loss = running_loss / len(data_loader)
    return avg_loss

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
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Start Training ---")
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs + 1):
        avg_val_loss = 0.0
        avg_train_loss = train_one_epoch(model, optimizer, train_data_loader, device, epoch, max_norm, print_freq)
        with torch.no_grad():
            avg_val_loss = validate_loss_one_epoch(model, val_data_loader, device, debug=False)
        lr_scheduler.step()
        print(f"Epoch {epoch} Done. \tAvg Training Loss: {avg_train_loss:.4f} \tAvg Validation Loss: {avg_val_loss:.4f}")
        # Save Checkpoint
        chkpt_path_config = config['paths']['model_checkpoint_prefix']
        chkpt_path = f"{chkpt_path_config}epoch_{epoch}.dat"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, chkpt_path)
        print(f"Saved checkpoint: {chkpt_path}")

    # Save final model for inference
    final_model_path = config['paths']['final_model_path']
    torch.save(model.state_dict(), final_model_path)

    print(f"Training finished in {time.time() - start_time:.0f} seconds.")

if __name__ == "__main__":
    main()