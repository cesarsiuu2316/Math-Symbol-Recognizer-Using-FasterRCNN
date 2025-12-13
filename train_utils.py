import torch
import bisect
from collections import defaultdict
from torch.utils.data.sampler import BatchSampler
from collections import Counter

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
        
        for idx in self.sampler:
            group_id = self.group_ids[idx].item()
            buffer_per_group[group_id].append(idx)
            
            # If this specific bucket is full, yield it
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                buffer_per_group[group_id] = []
        
        # Yield any remaining partial batches from all groups
        for group_id, indices in buffer_per_group.items():
            if len(indices) > 0:
                yield indices

    def __len__(self):
        return len(self.sampler) // self.batch_size

def create_aspect_ratio_groups(dataset):
    """
    Groups images into 3 Groups:
    0: Tall/Square (Ratio < 1.5)
    1: Wide        (1.5 <= Ratio < 3.0)
    2: Very Wide   (Ratio >= 3.0)
    Args:
        dataset (MathSymbolDataset): The dataset containing annotations with width and height.
    Returns:
        list: A list of group IDs corresponding to each image in the dataset.
    """
    print("Grouping images by Aspect Ratio (3 Groups)...")
    
    target_min_size = dataset.target_min_size
    scaling_factor = dataset.scaling_factor
    
    # --- CONFIGURATION: Aspect ratio breakpoints ---
    # Everything < 1 goes to Group 0
    # Everything 1 to 1.5 goes to Group 1
    # Everything 1.5 to 2 goes to Group 2
    # Everything >= 2 goes to Group 3
    breakpoints = [1.1, 1.5, 2] 
    group_ids = []
    
    for i in range(len(dataset)):
        # 1. Get Params
        item = dataset.annotations[i]
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