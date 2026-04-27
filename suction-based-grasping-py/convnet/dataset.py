import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class SuctionGraspingDataset(Dataset):
    """
    Dataset for suction grasping that loads color, depth, and label images.
    Preprocessing mirrors the original Torch/Lua implementation.
    """

    def __init__(
            self,
            data_path: str,
            sample_list: str,  # list of samples included in this train/test set
            output_scale: int,  # input image size / label image size
            img_height: int,
            img_width: int):
        super().__init__()
        self.data_path = data_path
        self.output_scale = output_scale
        self.img_height = img_height
        self.img_width = img_width

        # Read sample file names (one per line, no extension)
        with open(sample_list, "r") as f:
            self.sample_paths = [line.strip() for line in f.readlines() if line.strip()]
        self.num_samples = len(self.sample_paths)

        # ImageNet normalisation constants
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):  # TODO do all transforms with transform function given as argument
        sample_name = self.sample_paths[index]

        # ---------- Color image ----------
        color_path = os.path.join(self.data_path, "color-input", f"{sample_name}.png")
        color_pil = Image.open(color_path).convert("RGB")
        color_tensor = F.to_tensor(color_pil)                     # CxHxW, [0,1]
        color_tensor = F.normalize(
            color_tensor, mean=self.mean, std=self.std
        )

        # ---------- Depth image ----------
        depth_path = os.path.join(self.data_path, "depth-input", f"{sample_name}.png")
        depth_pil = Image.open(depth_path)                        # 16-bit single channel
        depth_np = np.array(depth_pil, dtype=np.float32)          # raw 0-65535
        # Replicate Lua scaling: loaded image is normalised to [0,1],
        # then multiplied by 65536/10000 → max ~6.5536, clamped to [0,1.2]
        depth_np = depth_np / 65535.0 * (65536.0 / 10000.0)
        depth_np = np.clip(depth_np, 0.0, 1.2)  # Depth range of Intel RealSense SR300
        # Stack three identical channels (original code used cat + reshape)
        depth_np = np.tile(depth_np[None, ...], (3, 1, 1))       # (3, H, W)
        depth_tensor = torch.from_numpy(depth_np)
        # Normalise each channel
        for c in range(3):
            depth_tensor[c] = (depth_tensor[c] - self.mean[c]) / self.std[c]

        # ---------- Label image ----------
        label_path = os.path.join(self.data_path, "label", f"{sample_name}.png")
        label_pil = Image.open(label_path)                        # grayscale
        label_tensor = F.to_tensor(label_pil)                     # 1xHxW, [0,1]
        label_tensor = torch.round(label_tensor * 2) + 1          # map to {1,2,3}
        # Downsample with nearest neighbour ('simple' in Lua)
        label_resized = F.resize(
            label_tensor,
            [self.img_height // self.output_scale, self.img_width // self.output_scale],
            interpolation=F.InterpolationMode.NEAREST,
        )
        label_resized = label_resized.squeeze(0)                  # (H_out, W_out)

        return color_tensor, depth_tensor, label_resized
