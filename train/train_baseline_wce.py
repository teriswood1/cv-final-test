"""
U-Net Baseline + 类别加权 CrossEntropy（同样使用修复后的数据增强划分）。
"""

from pathlib import Path
import sys

import torch
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.unet import UNet
from utils.data_split import create_train_val_loaders, verify_augment_flags
from utils.experiment import experiment_name_with_epochs
from utils.losses import (
    build_criterion,
    compute_class_pixel_counts,
    counts_to_weights,
)
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

    exp_name = experiment_name_with_epochs("unet_baseline_wce", num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("U-Net Baseline + Weighted CE")
    print("=" * 80)
    print(f"实验名：{exp_name}")
    print(f"使用设备：{device}")
    print("=" * 80)

    data_root = PROJECT_ROOT / "data" / "raw"
    train_image_dir = data_root / "train" / "images"
    train_label_dir = data_root / "train" / "labels"

    output_dir = PROJECT_ROOT / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = checkpoint_dir / f"{exp_name}_best.pth"
    last_model_path = checkpoint_dir / f"{exp_name}_last.pth"
    log_path = log_dir / f"{exp_name}_train_log.txt"
    weights_path = output_dir / "class_weights.pt"

    train_loader, val_loader, info = create_train_val_loaders(
        image_dir=train_image_dir,
        label_dir=train_label_dir,
        image_size=image_size,
        batch_size=batch_size,
        val_ratio=val_ratio,
        random_seed=random_seed,
        device_type=device.type,
    )

    verify_augment_flags(train_loader.dataset, val_loader.dataset)

    print(f"训练：{info['train_size']} augment={info['train_augment']}")
    print(f"验证：{info['val_size']} augment={info['val_augment']}")

    train_indices = train_loader.dataset.indices
    counts = compute_class_pixel_counts(
        label_dir=train_label_dir,
        image_dir=train_image_dir,
        indices=train_indices,
        num_classes=num_classes,
    )
    class_weights = counts_to_weights(counts, num_classes=num_classes)

    torch.save(
        {
            "counts": counts,
            "weights": class_weights,
            "num_classes": num_classes,
            "random_seed": random_seed,
        },
        weights_path,
    )

    print("类别权重：")
    for i in range(num_classes):
        print(f"  class_{i}: weight={class_weights[i].item():.4f}")

    model = UNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=base_channels,
    ).to(device)

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
        best_model_path=best_model_path,
        last_model_path=last_model_path,
        log_path=log_path,
        metric_class=SegmentationMetric,
        checkpoint_extra={
            "model_name": exp_name,
            "base_channels": base_channels,
            "loss": "weighted_ce",
            "augment_fix": True,
            "class_weights": class_weights.cpu(),
        },
    )


if __name__ == "__main__":
    main()
