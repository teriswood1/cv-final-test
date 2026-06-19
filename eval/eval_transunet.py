"""Evaluate the trained TransUNet checkpoint on data/raw/test."""

from pathlib import Path
import csv
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.transunet import TransUNet
from utils.dataset import UAVSegmentationDataset
from utils.experiment import experiment_name_with_epochs
from utils.metrics import SegmentationMetric
from utils.train_engine import evaluate


def save_results_txt(save_path, test_loss, pixel_acc, miou, class_iou):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("TransUNet Test Results (CNN + Transformer Fusion)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Pixel Accuracy: {pixel_acc:.6f}\n")
        f.write(f"Mean IoU: {miou:.6f}\n\nClass IoU:\n")
        for i, iou in enumerate(class_iou):
            f.write(f"class_{i}: {iou:.6f}\n")


def save_results_csv(save_path, test_loss, pixel_acc, miou, class_iou):
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Test Loss", f"{test_loss:.6f}"])
        writer.writerow(["Pixel Accuracy", f"{pixel_acc:.6f}"])
        writer.writerow(["Mean IoU", f"{miou:.6f}"])
        writer.writerow([])
        writer.writerow(["Class", "IoU"])
        for i, iou in enumerate(class_iou):
            writer.writerow([f"class_{i}", f"{iou:.6f}"])


def main():
    num_classes = 12
    image_size = (360, 480)
    batch_size = 2
    base_channels = 32
    transformer_dim = 256
    num_heads = 4
    num_layers = 2
    mlp_dim = 1024
    dropout = 0.1
    num_epochs = 50
    exp_name = experiment_name_with_epochs("transunet", num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = PROJECT_ROOT / "data" / "raw"
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"{exp_name}_best.pth"
    output_metric_dir = PROJECT_ROOT / "outputs" / "metrics"
    output_metric_dir.mkdir(parents=True, exist_ok=True)
    txt_save_path = output_metric_dir / f"{exp_name}_test_results.txt"
    csv_save_path = output_metric_dir / f"{exp_name}_test_results.csv"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}\n"
            "Run `python train_transunet.py` first."
        )

    test_dataset = UAVSegmentationDataset(
        image_dir=data_root / "test" / "images",
        label_dir=data_root / "test" / "labels",
        image_size=image_size,
        augment=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

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

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print("=" * 80)
    print(f"Testing: {exp_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch')}  Val mIoU: {checkpoint.get('miou')}")
    print(f"Test images: {len(test_dataset)}  Device: {device}")
    print("=" * 80)

    test_loss, pixel_acc, miou, class_iou = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        num_classes=num_classes,
        metric_class=SegmentationMetric,
    )

    print(f"Test Loss:      {test_loss:.6f}")
    print(f"Pixel Accuracy: {pixel_acc:.6f}")
    print(f"Mean IoU:       {miou:.6f}")
    save_results_txt(txt_save_path, test_loss, pixel_acc, miou, class_iou)
    save_results_csv(csv_save_path, test_loss, pixel_acc, miou, class_iou)
    print(f"TXT results: {txt_save_path}")
    print(f"CSV results: {csv_save_path}")


if __name__ == "__main__":
    main()
