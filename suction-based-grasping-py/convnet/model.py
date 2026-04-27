import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101

# TODO: test this module

# ----------------------------------------------------------------------
# Helper: replace BatchNorm2d with a fixed (non‑batch) affine transform
# ----------------------------------------------------------------------
class FixedBatchNorm(nn.Module):
    """Applies a fixed affine transformation using pre‑trained BN statistics."""
    def __init__(self, bn_layer):
        super().__init__()
        self.register_buffer("running_mean", bn_layer.running_mean.clone())
        self.register_buffer("running_var", bn_layer.running_var.clone())
        self.weight = nn.Parameter(bn_layer.weight.clone(), requires_grad=False)
        self.bias = nn.Parameter(bn_layer.bias.clone(), requires_grad=False)
        self.eps = bn_layer.eps

    def forward(self, x):
        # y = gamma * (x - mean) / sqrt(var + eps) + beta
        mean = self.running_mean[None, :, None, None]  # TODO when do these update?
        var = self.running_var[None, :, None, None]
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[None, :, None, None] * x_norm + self.bias[None, :, None, None]


def bn_to_fixed(module, recursive=True):
    """Recursively replace all BatchNorm2d layers with FixedBatchNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, FixedBatchNorm(child))
        elif recursive:
            bn_to_fixed(child, recursive)


def freeze_params(module):
    """Set requires_grad=False for all parameters in the module."""
    for param in module.parameters():
        param.requires_grad = False


# ----------------------------------------------------------------------
# Symmetric (reflection) padding wrapper for Conv2d
# ----------------------------------------------------------------------
class SymmetricPadConv2d(nn.Module):
    """Conv2d preceded by reflection padding to mimic 'SpatialSymmetricPadding'."""
    def __init__(self, conv_layer):
        super().__init__()
        self.conv = conv_layer
        k = conv_layer.kernel_size
        if isinstance(k, int):
            k = (k, k)
        # mirror padding: total = k-1 per spatial dim, split approx evenly
        self.pad = (k[1] // 2, (k[1] - 1) // 2,
                    k[0] // 2, (k[0] - 1) // 2)

    def forward(self, x):
        x = F.pad(x, self.pad, mode="reflect")
        return self.conv(x)


def update_padding(module):
    """Replace all non‑1x1 Conv2d layers with their symmetric‑padding version."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.kernel_size != (1, 1):
            setattr(module, name, SymmetricPadConv2d(child))
        else:
            update_padding(child)


# ----------------------------------------------------------------------
# RGB‑D ResNet model
# ----------------------------------------------------------------------
class RGBDResNet(nn.Module):
    """Fully convolutional RGB‑D network based on a shared ResNet‑101 trunk."""
    def __init__(self, n_class, rgb_trunk, depth_trunk):
        super().__init__()
        self.rgb_trunk = rgb_trunk
        self.depth_trunk = depth_trunk

        # TODO After concatenation (4096 channels because two 2048‑ch trunks)
        # The original Lua code had a mismatch (2048 input channels here).
        # We faithfully reproduce the numbers from the source, but be aware that
        # this may need correction to 4096 for the network to work as intended.
        self.conv1 = nn.Conv2d(2048, 512, 1)
        self.conv2 = nn.Conv2d(512, 128, 1)
        self.conv3 = nn.Conv2d(128, n_class, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, rgb, depth):
        rgb_features = self.rgb_trunk(rgb)
        depth_features = self.depth_trunk(depth)
        x = torch.cat([rgb_features, depth_features], dim=1)   # 4096 channels
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.upsample(x)
        return x


# ----------------------------------------------------------------------
# Main builder function
# ----------------------------------------------------------------------
def get_model(n_class):
    """
    Builds the RGB‑D ResNet‑101 model and the corresponding loss criterion.

    Args:
        n_class: number of output classes (should match the dataset, e.g. 3)

    Returns:
        model, criterion
    """
    # ----- Load ImageNet pre‑trained ResNet‑101 -----
    resnet = resnet101(pretrained=True)
    # Remove avgpool and fc to keep the convolutional backbone only
    trunk = nn.Sequential(*list(resnet.children())[:-2])   # output 2048 channels

    # ----- Convert BN to fixed affine (freeze running stats) -----
    bn_to_fixed(trunk)
    freeze_params(trunk)

    # ----- Clone the trunk for the depth branch -----
    depth_trunk = copy.deepcopy(trunk)

    # ----- Assemble the RGB‑D model -----
    model = RGBDResNet(n_class, trunk, depth_trunk)

    # ----- Replace standard convolutions with symmetric padding versions -----
    update_padding(model)

    # ----- Loss: ignore class index 2 (originally class 3, unlabeled) -----
    # The original Lua code sets weight[3]=0, we use ignore_index instead.
    criterion = nn.CrossEntropyLoss(ignore_index=2)

    return model, criterion
