from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

class InferenceDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img_tensor = read_image(path, mode=ImageReadMode.RGB) # Convert to Tensor [C, H, W]
        img_tensor = img_tensor.float() / 255.0 # Convert to [0, 1] range
        return img_tensor, path

def collate_fn(batch):
    return tuple(zip(*batch))