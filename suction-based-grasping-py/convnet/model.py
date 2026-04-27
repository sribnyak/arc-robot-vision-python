import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet101, ResNet101_Weights
from pathlib import Path


def freeze_batch_norm(model: nn.Module, disable_grad: bool = True) -> None:
    """
    Convert all BatchNorm layers to fixed (frozen) layers.
    
    Args:
        model: The model to modify in-place
        disable_grad: Disable gradients for BatchNorm parameters
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
            num_classes: Number of output classes
            pretrained: Whether to load ImageNet pretrained weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load ResNet-101 pre-trained on ImageNet
        resnet_weights = ResNet101_Weights.DEFAULT if pretrained else None
        resnet = resnet101(weights=resnet_weights)
        
        # Freeze BatchNorm layers [TODO: calling .eval() is probably bad]
        freeze_batch_norm(resnet)
        
        # Remove FC layers (avgpool and fc) - keep only convolutional trunk
        self.rgb_trunk = nn.Sequential(*list(resnet.children())[:-3])  # TODO: maybe 2 or 4
        
        # Create depth branch by cloning RGB trunk
        self.depth_trunk = copy.deepcopy(self.rgb_trunk)
        
        # Refinement head: process concatenated RGB-D features
        self.head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # TODO Replace zero padding with symmetric padding for 3x3 and 7x7 convolutions?
        # update_padding(self.head, SpatialSymmetricPadding)
    
    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RGB-D network.
        
        Args:
            rgb: RGB input tensor of shape (batch, 3, height, width)
            depth: Depth input tensor of shape (batch, 3, height, width)
            
        Returns:
            Classification/segmentation map (logits) of shape (batch, num_classes, height/16, width/16)
        """
        rgb_features = self.rgb_trunk(rgb)        # (batch, 1024, h/16, w/16)
        depth_features = self.depth_trunk(depth)  # (batch, 1024, h/16, w/16)
        
        combined_features = torch.cat([rgb_features, depth_features], dim=1)  # (batch, 2048, h/16, w/16)
        
        logit_map = self.head(combined_features)  # (batch, num_classes, h/8, w/8)
        
        return logit_map


# TODO move function to another file
def masked_bce_loss(logit_map: torch.Tensor, target: torch.Tensor, ignore_class: int = 2) -> torch.Tensor:
    """
    Calculate cross-entropy loss criterion, ignoring specified class.
    
    Args:
        logit_map: Predicted pixel-wise logit map, a tensor of shape (batch, 1, height, width)
        target: Target segmentation map, an integer-valued tensor of shape (batch, 3, height, width)
        ignore_class: Index of the class that will have weight 0.
        
    Returns:
        Tensor loss value
    """

    logit_map = logit_map.squeeze(1)  # (batch, h, w)
    valid_mask = (target != ignore_class)

    if valid_mask.any():
        valid_logits = logit_map[valid_mask]  # (batch,)
        valid_targets = target[valid_mask]    # (batch,)

        return F.binary_cross_entropy_with_logits(valid_logits, valid_targets.float())
    else:  # no valid pixels in batch
        return torch.tensor(0., device=logit_map.device, requires_grad=True)
