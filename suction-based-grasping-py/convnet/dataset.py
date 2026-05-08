"""
Module for loading and preprocessing the suction grasping dataset.
"""

import os
from typing import Any, Callable

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class DatasetInfo:
    """
    Constants connected to the dataset and the preprocessing pipeline.
    """

    # ImageNet normalisation constants
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Input image size / label image size
    output_scale = 8


class SuctionGraspingDataset(Dataset):
    """
    Dataset for suction grasping. Consists of triples (color image, depth map, label).
    Label is a grayscale image with values {0, 0.5, 1}, where 0 indicates positions
    not suitable for grasping, 0.5 - good positions for grasping, and 1 means that the
    quality of a grasp at this position is unknown and should not be used for training.
    """

    def __init__(
        self,
        data_path: str,
        sample_list: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        """Initialize the dataset class. Optionally set a transform to be applied to
        each data item.

        Args:
            data_path (str): path to the dataset.
            sample_list (str): path to the text file with the list of samples included
                in this train/test set.
            transform (Callable): the transform to be applied to color and depth input.
                By default - no transform.
            target_transform (Callable): the transform to be applied to the label image.
                By default - no transform.
        """
        super().__init__()

        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform

        # Read sample file names (one per line, no extension)
        with open(sample_list, "r") as f:
            self.sample_paths = [line.strip() for line in f.readlines() if line.strip()]

        self.num_samples = len(self.sample_paths)

    def __len__(self) -> int:
        """Return the number of data items in this dataset.

        Returns:
            int: the number of data items in the dataset.
        """
        return self.num_samples

    def __getitem__(self, index: int) -> Any:
        """Get one data item (color_image, depth_image, label_image) by index, according
        to the sample list, where each images is in PIL format. Apply transforms,
        if defined.

        Args:
            index (int): index of data item in the dataset.

        Returns:
            Any: returns a tuple of 3 PIL images: one RGB and two single-channel images,
                 with transforms applied, if defined.
        """
        sample_name = self.sample_paths[index]

        color_path = os.path.join(self.data_path, "color-input", f"{sample_name}.png")
        color = Image.open(color_path).convert("RGB")

        depth_path = os.path.join(self.data_path, "depth-input", f"{sample_name}.png")
        depth = Image.open(depth_path)  # 16-bit single channel

        label_path = os.path.join(self.data_path, "label", f"{sample_name}.png")
        label = Image.open(label_path)  # grayscale

        if self.transform:
            color, depth = self.transform(color, depth)
        if self.target_transform:
            label = self.target_transform(label)
        return color, depth, label


def data_transform(
    color_pil: Image, depth_pil: Image
) -> tuple[torch.Tensor, torch.Tensor]:
    """Data preprocessing, mirrors the original Torch/Lua implementation.
    Transforms images into PyTorch tensors of required format. Values are normalized,
    `depth_pil` is cloned across 3 channels. See the exact algorithm in code.

    Args:
        color_pil (Image): color image.
        depth_pil (Image): depth image (1 channel).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: the transformed images, two float-valued
            tensors of shape (3, H, W).
    """
    color_tensor = F.to_tensor(color_pil)  # shape (3, H, W), values [0, 1]
    color_tensor = F.normalize(color_tensor, mean=DatasetInfo.mean, std=DatasetInfo.std)

    depth_np = np.array(depth_pil, dtype=np.float32)  # raw 0-65535
    # Replicate Lua scaling: loaded image is normalised to [0, 1],
    # then multiplied by 65536/10000 -> max ~6.5536, clamped to [0, 1.2]
    depth_np = depth_np / 65535.0 * (65536.0 / 10000.0)
    depth_np = np.clip(depth_np, 0.0, 1.2)  # Depth range of Intel RealSense SR300

    # Stack three identical channels
    depth_np = np.tile(depth_np[None, ...], (3, 1, 1))  # (3, H, W)
    for c in range(3):
        depth_np[c] = (depth_np[c] - DatasetInfo.mean[c]) / DatasetInfo.std[c]
    depth_tensor = torch.from_numpy(depth_np)

    return color_tensor, depth_tensor


def target_transform(label_pil: Image) -> torch.Tensor:
    """Target preprocessing, mirrors the original Torch/Lua implementation.
    Transforms the image into a PyTorch tensor, resizes to match the shape of
    the network's output, the values are mapped to integers {0, 1, 2}. See the exact
    algorithm in code.

    Args:
        label_pil (Image): label image (1 channel).

    Returns:
        torch.Tensor: the transformed target image, a 2-dimensional tensor of integers.
    """
    label_tensor = F.to_tensor(label_pil)  # shape (1, H, W), values [0, 1]

    _, height, width = label_tensor.shape
    label_resized = F.resize(
        label_tensor,
        [height // DatasetInfo.output_scale, width // DatasetInfo.output_scale],
        interpolation=F.InterpolationMode.NEAREST,
    )
    label_resized = label_resized.squeeze(0)  # (H_out, W_out)

    # Map to {0, 1, 2} (2 = unknown)
    return torch.round(label_resized * 2).long()
