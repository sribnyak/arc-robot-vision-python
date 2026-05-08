"""
Model evaluation. Can be called from command line or imported.
"""

import hydra
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur
from tqdm.auto import tqdm

from dataset import SuctionGraspingDataset, data_transform, target_transform
from metrics import masked_bce_loss, masked_mse_loss
from model import RGBDResNet101


def double_target_transform(label_pil: Image) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return label image both raw (transformed into a tensor) and preprocessed.
    """
    return TF.to_tensor(label_pil), target_transform(label_pil)


def evaluate_model(
    model: RGBDResNet101,
    dataset: SuctionGraspingDataset,
    device: str | torch.device,
    batch_size: int = 4,
) -> dict[str, float]:
    """Calculate loss and quality metrics (masked cross-entropy, MSE, precision after
    filtering by threshold) for the given model on the given dataset. The dataset will
    be changed: its `target_transform` will be set to `double_target_transform`.

    Args:
        model (RGBDResNet101): the model with weights ready.
        dataset (SuctionGraspingDataset): the dataset (train/test/etc).
        device (str | torch.device): the device for calculating metrics.
        batch_size (int, optional): batch size for calculating metrics. Defaults to 4.

    Returns:
        Dict[str, float]: a dictionary with calculated metrics (metric: value).
    """

    dataset.target_transform = double_target_transform
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    gauss = GaussianBlur(kernel_size=15, sigma=7.0)  # adjust kernel/sigma if needed

    loss_sum = 0
    mse_sum = 0
    n = len(dataset)

    sum_tp = 0
    sum_fp = 0
    sum_tn = 0
    sum_fn = 0

    model.eval()
    for color_batch, depth_batch, (raw_label_batch, label_batch) in tqdm(data_loader):
        color_batch = color_batch.to(device)
        depth_batch = depth_batch.to(device)
        label_batch = label_batch.to(device)
        raw_label_batch = torch.round(raw_label_batch * 2).long().to(device)

        with torch.no_grad():
            logits = model(color_batch, depth_batch)  # (N, 1, H_out, W_out)
            loss = masked_bce_loss(logits, label_batch).cpu().item()
            output = F.sigmoid(logits)
            mse = masked_mse_loss(output, label_batch).cpu().item()

        batch_size = output.shape[0]
        loss_sum += loss * batch_size
        mse_sum += mse * batch_size

        # Resize output to match label shape
        *_, h_label, w_label = raw_label_batch.shape
        afford_map = F.interpolate(
            output, size=(h_label, w_label), mode="bilinear", align_corners=False
        )  # (N, 1, H, W)

        # Clamp as MATLAB did
        afford_map = afford_map.clamp(0.0, 0.9999)

        # Optional depth hole filling / postprocessing omitted

        # Gaussian smooth affordances (same as MATLAB)
        afford_map = gauss(afford_map)  # (N, 1, H, W)

        flat = afford_map.view(batch_size, -1)
        # Per-sample thresholds (top-1 pixel)
        thresholds = flat.max(dim=1).values - 1e-4  # (N,)
        # thresholds = 0.5  # Confidence threshold based
        # thresholds = percentile(afford_map, 99)  # Top 1%

        # Broadcast thresholds to score shape and compute pos mask
        thresholds = thresholds.view(batch_size, 1, 1, 1)
        pos_mask = afford_map > thresholds  # (N, 1, H, W), bool
        gt_pos = raw_label_batch == 1  # (N, 1, H, W)
        gt_neg = raw_label_batch == 0  # (N, 1, H, W)

        # # Only count pixels where label is valid (0 or 1)
        # valid_mask = raw_label_batch != 2
        # pos_mask = pos_mask & valid_mask

        # Compute TP/FP/TN/FN per batch and accumulate
        sum_tp += int((pos_mask & gt_pos).sum().item())
        sum_fp += int((pos_mask & gt_neg).sum().item())
        sum_tn += int((~pos_mask & gt_neg).sum().item())
        sum_fn += int((~pos_mask & gt_pos).sum().item())

    precision = sum_tp / (sum_tp + sum_fp)

    return {"bce_loss": loss_sum / n, "mse": mse_sum / n, "precision": precision}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def evaluate_cfg(cfg: DictConfig):
    """Evaluate model on test data using config parameters. Entrypoint for CLI.

    Args:
        cfg (DictConfig): project's config.
    """

    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {cfg.best_weights}")
    try:
        model = RGBDResNet101(num_classes=1, pretrained=False)
        checkpoint = torch.load(
            cfg.best_weights, map_location=device, weights_only=True
        )
        if "optimizer" in checkpoint and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print("Couldn't load weights. Using weights from pretraining on ImageNet")
        model = RGBDResNet101(num_classes=1, pretrained=True)
    model = model.to(device).eval()

    dataset = SuctionGraspingDataset(
        data_path=cfg.data_path, sample_list=cfg.test_split, transform=data_transform
    )

    results = evaluate_model(model, dataset, device, cfg.eval_batch_size)
    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")


if __name__ == "__main__":
    evaluate_cfg()
