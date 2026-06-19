"""Train the TransUNet fusion model for phase four."""

from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.transunet import TransUNet
from utils.data_split import create_train_val_loaders, verify_augment_flags
from utils.experiment import experiment_name_with_epochs
from utils.metrics import SegmentationMetric
from utils.train_engine import run_training


def main():
    num_classes = 12
    image_size = (360, 480)
    base_channels = 32
    transformer_dim = 256
    num_heads = 4
    num_layers = 2
    mlp_dim = 1024
    dropout = 0.1

    batch_size = 2
    num_epochs = 50
    learning_rate = 1e-4
    val_ratio = 0.2
    random_seed = 42
    exp_name = experiment_name_with_epochs("transunet", num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("TransUNet Training (CNN + Transformer Fusion)")
    print("=" * 80)
    print(f"Experiment: {exp_name}")
    print(f"Device: {device}")
    print(f"Image size: {image_size}  batch_size: {batch_size}")
    print(f"Epochs: {num_epochs}  lr: {learning_rate}")
    print(
        "Transformer: "
        f"dim={transformer_dim}, heads={num_heads}, layers={num_layers}, mlp={mlp_dim}"
    )
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

    print(f"Total images: {info['total']}")
    print(f"Train: {info['train_size']}  augment={info['train_augment']}")
    print(f"Val:   {info['val_size']}  augment={info['val_augment']}")

    model = TransUNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=base_channels,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    start_epoch = 1
    best_miou = 0.0
    append_log = False
    if last_model_path.exists():
        checkpoint = torch.load(
            last_model_path,
            map_location=device,
            weights_only=False,
        )
        last_epoch = int(checkpoint.get("epoch", 0))
        if last_epoch >= num_epochs:
            print(f"Existing checkpoint already reached epoch {last_epoch}.")
            print(f"Best checkpoint: {best_model_path}")
            return

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = last_epoch + 1
        append_log = log_path.exists()

        if best_model_path.exists():
            best_checkpoint = torch.load(
                best_model_path,
                map_location=device,
                weights_only=False,
            )
            best_miou = float(best_checkpoint.get("miou", 0.0))
        else:
            best_miou = float(checkpoint.get("miou", 0.0))

        print(
            f"Resuming from epoch {start_epoch}/{num_epochs}; "
            f"current best mIoU={best_miou:.6f}"
        )

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
        start_epoch=start_epoch,
        best_miou=best_miou,
        append_log=append_log,
        checkpoint_extra={
            "model_name": exp_name,
            "model_family": "transunet",
            "base_channels": base_channels,
            "transformer_dim": transformer_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "mlp_dim": mlp_dim,
            "dropout": dropout,
            "loss": "ce",
            "augment_fix": True,
            "num_epochs": num_epochs,
        },
    )


if __name__ == "__main__":
    main()
