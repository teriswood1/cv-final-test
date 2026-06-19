"""U-Net + CBAM + ASPP（修复数据增强后训练，输出 unet_cbam_aspp_augfix_*）。"""
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.unet_cbam_aspp import UNetCBAMASPP
from utils.data_split import create_train_val_loaders, verify_augment_flags
from utils.experiment import experiment_name_with_epochs
from utils.metrics import SegmentationMetric
from utils.train_engine import run_training


def main():
    num_classes = 12
    image_size = (360, 480)
    base_channels = 32
    batch_size = 2
    num_epochs = 50
    learning_rate = 1e-4
    val_ratio = 0.2
    random_seed = 42
    exp_name = experiment_name_with_epochs("unet_cbam_aspp_augfix", num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("U-Net + CBAM + ASPP Training (Augmentation Fix)")
    print("=" * 80)

    data_root = PROJECT_ROOT / "data" / "raw"
    output_dir = PROJECT_ROOT / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, info = create_train_val_loaders(
        image_dir=data_root / "train" / "images",
        label_dir=data_root / "train" / "labels",
        image_size=image_size,
        batch_size=batch_size,
        val_ratio=val_ratio,
        random_seed=random_seed,
        device_type=device.type,
    )
    verify_augment_flags(train_loader.dataset, val_loader.dataset)
    print(f"训练 {info['train_size']} augment={info['train_augment']}")
    print(f"验证 {info['val_size']} augment={info['val_augment']}")

    model = UNetCBAMASPP(3, num_classes, base_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_classes=num_classes,
        num_epochs=num_epochs,
        best_model_path=checkpoint_dir / f"{exp_name}_best.pth",
        last_model_path=checkpoint_dir / f"{exp_name}_last.pth",
        log_path=log_dir / f"{exp_name}_train_log.txt",
        metric_class=SegmentationMetric,
        checkpoint_extra={
            "model_name": exp_name,
            "base_channels": base_channels,
            "loss": "ce",
            "augment_fix": True,
        },
    )


if __name__ == "__main__":
    main()
