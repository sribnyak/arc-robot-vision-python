import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np

from dataset import data_transform
from model import RGBDResNet101


parser = argparse.ArgumentParser(description='Model inference')
parser.add_argument('--rgb', type=str, default='demo/test-image.color.png', help='Path to RGB image')
parser.add_argument('--depth', type=str, default='demo/test-image.depth.png', help='Path to depth image')
parser.add_argument('--weights', type=str, default='weights/default_weights.pth', help='Path to model weights')
parser.add_argument('--output', type=str, default='results.png', help='Path to output png file')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model
print(f"Loading model: {args.weights}")
try:
    model = RGBDResNet101(num_classes=1, pretrained=False)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
except Exception:
    print("Couldn't load weights. Using weights from pretraining on ImageNet")
    model = RGBDResNet101(num_classes=1, pretrained=True)
model = model.to(device).eval()

# Load and preprocess images
color_pil = Image.open(args.rgb).convert("RGB")
depth_pil = Image.open(args.depth)

color, depth, _ = data_transform(color_pil, depth_pil, np.zeros_like(depth_pil))  # TODO

print(f"RGB shape: {color.shape}, Depth shape: {depth.shape}")

# Run inference
print("Computing forward pass...")
with torch.no_grad():
    color = color[None].to(device)
    depth = depth[None].to(device)
    output = F.sigmoid(model(color, depth.to(device)))

print(f"Output shape: {output.shape}")

# Save results
print(f"Saving results to: {args.output}")
Path(args.output).parent.mkdir(parents=True, exist_ok=True)

output_np = output.cpu().squeeze().numpy()
Image.fromarray((output_np * 255).astype(np.uint8), mode='L').save(args.output)
