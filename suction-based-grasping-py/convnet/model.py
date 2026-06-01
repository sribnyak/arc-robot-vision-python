"""
The RGB-D ResNet-101 neural network.
"""

import copy
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet101_Weights, resnet101


def freeze_batch_norm(model: nn.Module, disable_grad: bool = True) -> None:
    """
    Convert all BatchNorm layers to fixed (frozen) layers.

    Args:
        model: the model to modify in-place.
        disable_grad: disable gradients for BatchNorm parameters.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
            if disable_grad:
                for param in module.parameters():
                    param.requires_grad = False


class RGBDResNet101(nn.Module):
    """
    Fully convolutional RGB-D ResNet-101 model.

    Takes RGB and depth inputs through separate ResNet-101 trunks,
    concatenates the features, and produces segmentation predictions.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        """
        Initialize RGB-D ResNet-101 model.

        Args:
            num_classes: number of output classes.
            pretrained: whether to load ImageNet pretrained weights.
        """
        super().__init__()

        self.num_classes = num_classes

        # Load ResNet-101 pre-trained on ImageNet
        resnet_weights = ResNet101_Weights.DEFAULT if pretrained else None
        resnet = resnet101(weights=resnet_weights)

        # Freeze BatchNorm layers [TODO: calling .eval() is probably bad]
        freeze_batch_norm(resnet)

        # Remove the last 3 layers (4 in original code), including FC classifier part
        self.rgb_trunk = nn.Sequential(*list(resnet.children())[:-3])

        # Create depth branch by cloning RGB trunk
        self.depth_trunk = copy.deepcopy(self.rgb_trunk)

        # Refinement head: process concatenated RGB-D features
        self.head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        # TODO Replace zero padding with symmetric padding for 3x3 and 7x7 convolutions?
        # update_padding(self.head, SpatialSymmetricPadding)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RGB-D network.

        Args:
            rgb: RGB input tensor of shape (batch, 3, height, width).
            depth: depth input tensor of shape (batch, 3, height, width).

        Returns:
            Classification/segmentation map (logits) of shape
                (batch, num_classes, height/16, width/16).
        """
        rgb_features = self.rgb_trunk(rgb)
        # (batch, 1024, h/16, w/16)
        depth_features = self.depth_trunk(depth)
        # (batch, 1024, h/16, w/16)
        combined_features = torch.cat([rgb_features, depth_features], dim=1)
        # (batch, 2048, h/16, w/16)
        logit_map = self.head(combined_features)
        # (batch, num_classes, h/8, w/8)
        return logit_map


def load_model(weights_path: str | Path, device: str | torch.device) -> RGBDResNet101:
    """Load the RGBDResNet101 model with specified weights. If weights are not found,
    use ImageNet pretrained backbone.
    """
    print(f"Loading model from {weights_path}")
    try:
        model = RGBDResNet101(num_classes=1, pretrained=False)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        if "optimizer" in checkpoint and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print("Weights not found - using ImageNet pretrained backbone.")
        model = RGBDResNet101(num_classes=1, pretrained=True)
    return model
