import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig

from dataset import SuctionGraspingDataset, data_transform, target_transform
from model import RGBDResNet101
from metrics import masked_bce_loss


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(cfg.seed)

    dataset = SuctionGraspingDataset(
        data_path=cfg.data_path,
        sample_list=cfg.train_split,
        transform=data_transform,
        target_transform=target_transform,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_data,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    model = RGBDResNet101(num_classes=cfg.num_classes, pretrained=True).to(device)
    criterion = masked_bce_loss
    optimizer = optim.SGD(
        model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=cfg.log_dir)
    Path(cfg.snapshots_folder).mkdir(parents=True, exist_ok=True)

    model.train()
    train_iter = 1
    while train_iter <= cfg.max_iterations:  # TODO epochs
        for color_batch, depth_batch, label_batch in data_loader:
            optimizer.zero_grad()
            output = model(color_batch.to(device), depth_batch.to(device))
            loss = criterion(output, label_batch.to(device))
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            print(f"Iteration {train_iter}: loss={loss_val:.6f}")

            writer.add_scalar("Loss/train", loss_val, train_iter)
            if train_iter % cfg.log_interval == 0:
                writer.flush()

            if train_iter % cfg.snapshot_interval == 0:
                checkpoint = {
                    "iteration": train_iter,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": loss_val,
                }
                path = f"{cfg.snapshots_folder}/snapshot-{train_iter}.pt"
                torch.save(checkpoint, path)
                print(f"Checkpoint saved: {path}")

            if train_iter == cfg.max_iterations:
                break
            train_iter += 1

    writer.close()
    print("Training completed")


if __name__ == "__main__":
    train()
