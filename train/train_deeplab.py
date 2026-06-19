"""Train the pretrained DeepLabV3 CamVid-style road-scene baseline."""

from dataclasses import asdict
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.deeplab import build_deeplabv3_resnet50
from utils.data_split import create_train_eval_loaders
from utils.experiment import experiment_name_with_epochs
from utils.label_protocol import CAMVID_12_ID_PROTOCOL
from utils.metrics import SegmentationMetric
from utils.train_engine import run_training


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def build_cross_entropy(ignore_index):
    if ignore_index is None:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(ignore_index=ignore_index)


def build_metric_class(ignore_index):
    return lambda num_classes: SegmentationMetric(
        num_classes=num_classes,
        ignore_index=ignore_index,
    )


def main():
    protocol = CAMVID_12_ID_PROTOCOL

    image_size = (360, 480)
    batch_size = 2
    num_epochs = 50
    learning_rate = 1e-4
    min_learning_rate = 1e-6
    weight_decay = 1e-4
    exp_name = experiment_name_with_epochs(
        "deeplabv3_resnet50_pretrained_road",
        num_epochs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = PROJECT_ROOT / "data" / "raw"
    train_image_dir = data_root / "train" / "images"
    train_label_dir = data_root / "train" / "labels"
    eval_image_dir = data_root / "test" / "images"
    eval_label_dir = data_root / "test" / "labels"

    output_dir = PROJECT_ROOT / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Pretrained DeepLabV3-ResNet50 Road-Scene Training")
    print("=" * 80)
    print(f"Experiment: {exp_name}")
    print(f"Device: {device}")
    print(f"Train split: {train_image_dir}")
    print(f"Current eval split: {eval_image_dir}")
    print("Split policy: use all train images; select best checkpoint on current eval split")
    print(f"Label protocol: {protocol}")
    print(f"Image size: {image_size}  batch_size: {batch_size}")
    print(f"Epochs: {num_epochs}  lr: {learning_rate}  min_lr: {min_learning_rate}")
    print(f"Weight decay: {weight_decay}  augmentation: road")
    print("=" * 80)

    train_loader, eval_loader, info = create_train_eval_loaders(
        train_image_dir=train_image_dir,
        train_label_dir=train_label_dir,
        eval_image_dir=eval_image_dir,
        eval_label_dir=eval_label_dir,
        image_size=image_size,
        batch_size=batch_size,
        device_type=device.type,
        train_augment_mode="road",
        num_label_ids=protocol.num_label_ids,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
    )
    print(
        f"Training images: {info['train_size']} "
        f"augment={info['train_augment']} mode={info['train_augment_mode']} "
        f"drop_last={info['train_drop_last']}"
    )
    print(f"Eval images: {info['eval_size']} augment={info['eval_augment']}")

    model = build_deeplabv3_resnet50(
        num_classes=protocol.train_num_classes,
        pretrained=True,
    ).to(device)
    criterion = build_cross_entropy(protocol.ignore_index)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=min_learning_rate,
    )

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=eval_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=protocol.train_num_classes,
        num_epochs=num_epochs,
        best_model_path=checkpoint_dir / f"{exp_name}_best.pth",
        last_model_path=checkpoint_dir / f"{exp_name}_last.pth",
        log_path=log_dir / f"{exp_name}_train_log.txt",
        metric_class=build_metric_class(protocol.ignore_index),
        checkpoint_extra={
            "model_name": exp_name,
            "model_family": "deeplabv3_resnet50",
            "pretrained": True,
            "loss": "ce",
            "label_protocol": asdict(protocol),
            "image_size": image_size,
            "image_mean": IMAGE_MEAN,
            "image_std": IMAGE_STD,
            "train_augment_mode": "road",
            "train_split_policy": "full_train_current_eval",
            "eval_split": str(eval_image_dir),
        },
    )


if __name__ == "__main__":
    main()
