import torch
from collections import defaultdict, Counter
from torch.utils.data import BatchSampler
from PIL import Image
from tqdm import tqdm

class GroupedBatchSampler(BatchSampler):
    """
    Groups indices by ID. Yields batches where all images belong to the 
    same group to minimize padding and resizing artifacts.
    """
    def __init__(self, sampler, group_ids, batch_size):
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        
        for idx in self.sampler:
            group_id = self.group_ids[idx].item()
            buffer_per_group[group_id].append(idx)
            
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                buffer_per_group[group_id] = []
        
        # Yield remaining
        for group_id, indices in buffer_per_group.items():
            if len(indices) > 0:
                yield indices

    def __len__(self):
        return len(self.sampler) // self.batch_size

def create_exact_dimension_groups(image_paths, verbose=True):
    """
    Groups images by their EXACT (Width, Height).
    Ensures batches contain identical image sizes.
    Args: 
        image_paths (list): List of image file paths.
        verbose (bool): Whether to print progress and stats.
    Returns:
        list: Group IDs corresponding to each image path.
    """
    if verbose:
        print("Grouping images by Exact Dimensions...")
    
    # Dictionary to map unique (w, h) tuples to a Group ID
    # Example: {(100, 200): 0, (500, 500): 1}
    dims_to_id = {}
    next_id = 0
    group_ids = []
    
    # Statistics for verbose output
    stats = Counter()

    for path in tqdm(image_paths, disable=not verbose, desc="Scanning Dimensions"):
        with Image.open(path) as img:
            w, h = img.size
            dims = (w, h)

        # Assign ID
        if dims not in dims_to_id:
            dims_to_id[dims] = next_id
            next_id += 1
        
        gid = dims_to_id[dims]
        group_ids.append(gid)
        stats[dims] += 1

    if verbose:
        print(f"Found {len(dims_to_id)} unique image dimensions.")
        # Sort stats by most common resolutions
        for dim, count in stats.most_common(10):
            print(f"Dimension {dim}: {count} images")
        
    return group_ids