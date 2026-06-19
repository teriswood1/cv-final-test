"""
U-Net + ASPP + Weighted CE 训练脚本。

在 fixed_aug 基础上加入类别加权，但对极端权重做裁剪，避免训练崩坏。
"""

from pathlib import Path
import sys

import torch
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.unet_aspp import UNetASPP
from utils.data_split import create_train_val_loaders, verify_augment_flags
from utils.experiment import experiment_name_with_epochs
from utils.losses import build_criterion, compute_class_pixel_counts, counts_to_weights
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
    weight_clip_max = 2.0
    exp_name = experiment_name_with_epochs("unet_aspp_weighted_ce", num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("U-Net + ASPP Training (Weighted CE, 30 Epochs)")
    print("=" * 80)
    print(f"实验名: {exp_name}")
    print(f"设备: {device}")

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
        train_augment_mode="light",
    )
    verify_augment_flags(train_loader.dataset, val_loader.dataset)
    print(
        f"训练 {info['train_size']} augment={info['train_augment']} mode={info['train_augment_mode']}"
    )
    print(f"验证 {info['val_size']} augment={info['val_augment']}")

    counts = compute_class_pixel_counts(
        label_dir=data_root / "train" / "labels",
        image_dir=data_root / "train" / "images",
        indices=train_loader.dataset.indices,
        num_classes=num_classes,
    )
    class_weights = counts_to_weights(
        counts,
        num_classes=num_classes,
        mode="median_freq",
        clip_max=weight_clip_max,
    )
    print(f"权重裁剪上限: {weight_clip_max}")
    for i in range(num_classes):
        print(f"  class_{i}: weight={class_weights[i].item():.4f}")

    model = UNetASPP(3, num_classes, base_channels).to(device)
    criterion = build_criterion(
        loss_name="weighted_ce",
        class_weights=class_weights,
        device=device,
    )
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
            "loss": "weighted_ce",
            "weight_clip_max": weight_clip_max,
            "class_weights": class_weights.cpu(),
            "augment_fix": True,
            "train_augment_mode": "light",
            "num_epochs": num_epochs,
        },
    )


if __name__ == "__main__":
    main()

