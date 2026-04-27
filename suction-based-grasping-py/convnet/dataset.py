import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class DatasetInfo:
    # ImageNet normalisation constants
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    output_scale = 8  # input image size / label image size


class SuctionGraspingDataset(Dataset):
    """
    Dataset for suction grasping that loads color, depth, and label images.
    Preprocessing mirrors the original Torch/Lua implementation.
    """

    def __init__(
            self,
            data_path: str,
            sample_list: str,  # list of samples included in this train/test set
            transform=None):
        super().__init__()

        self.data_path = data_path
        self.transform = transform

        # Read sample file names (one per line, no extension)
        with open(sample_list, "r") as f:
            self.sample_paths = [line.strip() for line in f.readlines() if line.strip()]

        self.num_samples = len(self.sample_paths)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        sample_name = self.sample_paths[index]

        color_path = os.path.join(self.data_path, "color-input", f"{sample_name}.png")
        color_pil = Image.open(color_path).convert("RGB")

        depth_path = os.path.join(self.data_path, "depth-input", f"{sample_name}.png")
        depth_pil = Image.open(depth_path)  # 16-bit single channel

        label_path = os.path.join(self.data_path, "label", f"{sample_name}.png")
        label_pil = Image.open(label_path)  # grayscale

        if self.transform:
            return self.transform(color_pil, depth_pil, label_pil)
        return color_pil, depth_pil, label_pil


def data_transform(color_pil, depth_pil, label_pil):
    color_tensor = F.to_tensor(color_pil)                     # CxHxW, [0, 1]
    color_tensor = F.normalize(color_tensor,
        mean=DatasetInfo.mean, std=DatasetInfo.std)

    depth_np = np.array(depth_pil, dtype=np.float32)          # raw 0-65535
    # Replicate Lua scaling: loaded image is normalised to [0, 1],
    # then multiplied by 65536/10000 → max ~6.5536, clamped to [0, 1.2]
    depth_np = depth_np / 65535.0 * (65536.0 / 10000.0)
    depth_np = np.clip(depth_np, 0.0, 1.2)  # Depth range of Intel RealSense SR300
    # Stack three identical channels
    depth_np = np.tile(depth_np[None, ...], (3, 1, 1))        # (3, H, W)
    for c in range(3):
        depth_np[c] = (depth_np[c] - DatasetInfo.mean[c]) / DatasetInfo.std[c]
    depth_tensor = torch.from_numpy(depth_np)

    label_tensor = F.to_tensor(label_pil)                     # 1xHxW, [0, 1]
    label_tensor = torch.round(label_tensor * 2).long()       # map to {0, 1, 2}
    _, height, width = color_tensor.shape
    label_resized = F.resize(label_tensor,
        [height // DatasetInfo.output_scale, width // DatasetInfo.output_scale],
        interpolation=F.InterpolationMode.NEAREST)
    label_resized = label_resized.squeeze(0)                  # (H_out, W_out)

    return color_tensor, depth_tensor, label_resized
