from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
import errno
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from lib.model import TinyLidarNet
from lib.tiny_lidar_imagenet import TinyLidarImageNet
from lib.data import MultiSeqConcatDataset
from lib.loss import WeightedSmoothL1Loss



def clean_numerical_tensor(x: torch.Tensor) -> torch.Tensor:
    """NaN, infを安全に除去"""
    if torch.isnan(x).any() or torch.isinf(x).any():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


@hydra.main(config_path="./config", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    print("------ Configuration ------")
    print(OmegaConf.to_yaml(cfg))
    print("---------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Dataset ===
    # Pass image-related options to datasets so they can return (scan, img, target)
    train_dataset = MultiSeqConcatDataset(
        cfg.data.train_dir,
        image_dir=cfg.data.get("image_dir", "images"),
        img_scan_ratio=cfg.data.get("img_scan_ratio", 4),
        img_size=cfg.data.get("img_size", 224),
        img_mean=tuple(cfg.data.get("img_mean", [0.485, 0.456, 0.406])),
        img_std=tuple(cfg.data.get("img_std", [0.229, 0.224, 0.225])),
    )

    val_dataset = MultiSeqConcatDataset(
        cfg.data.val_dir,
        image_dir=cfg.data.get("image_dir", "images"),
        img_scan_ratio=cfg.data.get("img_scan_ratio", 4),
        img_size=cfg.data.get("img_size", 224),
        img_mean=tuple(cfg.data.get("img_mean", [0.485, 0.456, 0.406])),
        img_std=tuple(cfg.data.get("img_std", [0.229, 0.224, 0.225])),
    )

    def make_dataloader(dataset, **dl_kwargs):
        """Create a DataLoader and if an OSError (No space / semaphore error)
        occurs, fall back to num_workers=0 and retry once.
        """
        try:
            return DataLoader(dataset, **dl_kwargs)
        except OSError as e:
            # errno 28 -> No space left on device (often when semaphores or /dev/shm are low)
            if getattr(e, "errno", None) == errno.ENOSPC or "No space left" in str(e):
                print(f"[WARN] DataLoader creation failed with OSError: {e}. Retrying with num_workers=0")
                dl_kwargs["num_workers"] = 0
                return DataLoader(dataset, **dl_kwargs)
            raise

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False
    )
    train_loader = make_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = make_dataloader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # === Model ===
    # === Model selection ===
    if cfg.model.name == "TinyLidarImageNet":
        # pass optional resnet options if present in config
        resnet_ckpt = cfg.model.get("resnet_checkpoint", None)
        resnet_pretrained = cfg.model.get("resnet_pretrained", False)
        freeze_resnet = cfg.model.get("freeze_resnet", False)
        resnet_type = cfg.model.get("resnet_type", "resnet15")
        resnet_out_dim = cfg.model.get("resnet_out_dim", 256)

        model = TinyLidarImageNet(
            input_dim=cfg.model.input_dim,
            output_dim=cfg.model.output_dim,
            resnet_pretrained=resnet_pretrained,
            resnet_checkpoint=resnet_ckpt,
            freeze_resnet=freeze_resnet,
            resnet_type=resnet_type,
            resnet_out_dim=resnet_out_dim,
        ).to(device)
        multimodal = True
    else:
        model = TinyLidarNet(
            input_dim=cfg.model.input_dim,
            output_dim=cfg.model.output_dim
        ).to(device)
        multimodal = False

    # Load pretrained weights (full model) if provided
    if cfg.train.pretrained_path:
        try:
            model.load_state_dict(torch.load(cfg.train.pretrained_path))
            print(f"[INFO] Loaded pretrained model from {cfg.train.pretrained_path}")
        except Exception:
            print(f"[WARN] Failed to load full model state from {cfg.train.pretrained_path}; continuing")

    # === Loss & Optimizer ===
    criterion = WeightedSmoothL1Loss(
        steer_weight=cfg.train.loss.steer_weight,
        accel_weight=cfg.train.loss.accel_weight
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)

    # === Logging & Save dirs ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(cfg.train.save_dir).expanduser().resolve()
    log_dir = Path(cfg.train.log_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with SummaryWriter(log_dir / timestamp) as writer:
        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = cfg.train.get("early_stop_patience", 10)

        best_path = save_dir / "best_model.pth"
        last_path = save_dir / "last_model.pth"

        # === Training Loop ===
        for epoch in range(cfg.train.epochs):
            model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{cfg.train.epochs}"):
                if multimodal:
                    scans, imgs, targets = batch
                    scans = scans.unsqueeze(1).to(device)
                    imgs = imgs.to(device)
                    targets = targets.to(device)

                    scans = clean_numerical_tensor(scans)
                    targets = clean_numerical_tensor(targets)

                    outputs = model(scans, imgs)
                else:
                    scans, targets = batch
                    scans = scans.unsqueeze(1).to(device)
                    targets = targets.to(device)

                    scans = clean_numerical_tensor(scans)
                    targets = clean_numerical_tensor(targets)

                    outputs = model(scans)

                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = validate(model, val_loader, device, criterion, multimodal=multimodal)

            print(f"Epoch {epoch+1:03d}: Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f}")
            writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
            writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_path)
                print(f"[SAVE] Best model updated: {best_path} (val_loss={best_val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            torch.save(model.state_dict(), last_path)
            if patience_counter >= max_patience:
                print(f"[EarlyStop] No improvement for {max_patience} epochs.")
                break
    
    print("Training finished.")


def validate(model, loader, device, criterion, multimodal: bool = False):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Val]", leave=False):
            if multimodal:
                scans, imgs, targets = batch
                scans = scans.unsqueeze(1).to(device)
                imgs = imgs.to(device)
                targets = targets.to(device)

                scans = clean_numerical_tensor(scans)
                targets = clean_numerical_tensor(targets)

                outputs = model(scans, imgs)
            else:
                scans, targets = batch
                scans = scans.unsqueeze(1).to(device)
                targets = targets.to(device)

                scans = clean_numerical_tensor(scans)
                targets = clean_numerical_tensor(targets)

                outputs = model(scans)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    main()
