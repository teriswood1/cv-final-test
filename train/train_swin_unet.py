"""Train ImageNet-pretrained Swin-T + FPN decoder for segmentation."""

from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.pretrained_swin_unet import SwinUNet
from utils.data_split import create_train_eval_loaders
from utils.experiment import experiment_name_with_epochs
from utils.metrics import SegmentationMetric
from utils.train_engine import run_training


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def main():
    num_classes = 12
    image_size = (360, 480)
    batch_size = 2
    num_epochs = 50
    encoder_lr = 1e-5
    decoder_lr = 1e-4
    min_learning_rate = 1e-6
    weight_decay = 1e-4
    decoder_channels = 128
    exp_name = experiment_name_with_epochs("swin_unet_pretrained", num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = PROJECT_ROOT / "data" / "raw"
    output_dir = PROJECT_ROOT / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = checkpoint_dir / f"{exp_name}_best.pth"
    last_model_path = checkpoint_dir / f"{exp_name}_last.pth"
    log_path = log_dir / f"{exp_name}_train_log.txt"

    print("=" * 80)
    print("Pretrained Swin-UNet Training")
    print("=" * 80)
    print(f"Experiment: {exp_name}")
    print(f"Device: {device}")
    print(f"Image size: {image_size}  batch_size: {batch_size}")
    print(f"Epochs: {num_epochs}  encoder_lr: {encoder_lr}  decoder_lr: {decoder_lr}")
    print("Split policy: use all train images; select best checkpoint on current eval split")
    print("=" * 80)

    train_loader, eval_loader, info = create_train_eval_loaders(
        train_image_dir=data_root / "train" / "images",
        train_label_dir=data_root / "train" / "labels",
        eval_image_dir=data_root / "test" / "images",
        eval_label_dir=data_root / "test" / "labels",
        image_size=image_size,
        batch_size=batch_size,
        device_type=device.type,
        train_augment_mode="road",
        num_label_ids=num_classes,
        image_mean=IMAGE_MEAN,
        image_std=IMAGE_STD,
    )
    print(
        f"Training images: {info['train_size']} "
        f"augment={info['train_augment']} mode={info['train_augment_mode']} "
        f"drop_last={info['train_drop_last']}"
    )
    print(f"Eval images: {info['eval_size']} augment={info['eval_augment']}")

    model = SwinUNet(
        num_classes=num_classes,
        pretrained=True,
        decoder_channels=decoder_channels,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.decoder.parameters(), "lr": decoder_lr},
        ],
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=min_learning_rate,
    )

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
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
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
        val_loader=eval_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
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
            "model_family": "swin_unet",
            "pretrained": True,
            "backbone": "swin_t",
            "decoder_channels": decoder_channels,
            "encoder_lr": encoder_lr,
            "decoder_lr": decoder_lr,
            "loss": "ce",
            "image_size": image_size,
            "image_mean": IMAGE_MEAN,
            "image_std": IMAGE_STD,
            "train_augment_mode": "road",
            "train_split_policy": "full_train_current_eval",
        },
    )


if __name__ == "__main__":
    main()
