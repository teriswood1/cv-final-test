"""Evaluate the pretrained DeepLabV3 CamVid-style road-scene checkpoint."""

import csv
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.deeplab import build_deeplabv3_resnet50
from utils.dataset import UAVSegmentationDataset
from utils.experiment import experiment_name_with_epochs
from utils.label_protocol import CAMVID_12_ID_PROTOCOL
from utils.metrics import SegmentationMetric


def build_cross_entropy(ignore_index):
    if ignore_index is None:
        return nn.CrossEntropyLoss()
    return nn.CrossEntropyLoss(ignore_index=ignore_index)


def evaluated_class_ids(num_classes, ignore_index):
    return [class_id for class_id in range(num_classes) if class_id != ignore_index]


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device, metric):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item()
        metric.update(outputs, labels)

    return total_loss / len(dataloader), metric.get_results()


def save_results_txt(save_path, test_loss, pixel_acc, miou, class_ids, class_iou):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("DeepLabV3-ResNet50 Current Eval Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Eval Loss: {test_loss:.6f}\n")
        f.write(f"Pixel Accuracy: {pixel_acc:.6f}\n")
        f.write(f"Mean IoU: {miou:.6f}\n")
        f.write("\nClass IoU:\n")
        for class_id, iou in zip(class_ids, class_iou):
            f.write(f"class_{class_id}: {iou:.6f}\n")


def save_results_csv(save_path, test_loss, pixel_acc, miou, class_ids, class_iou):
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Eval Loss", f"{test_loss:.6f}"])
        writer.writerow(["Pixel Accuracy", f"{pixel_acc:.6f}"])
        writer.writerow(["Mean IoU", f"{miou:.6f}"])
        writer.writerow([])
        writer.writerow(["Class", "IoU"])
        for class_id, iou in zip(class_ids, class_iou):
            writer.writerow([f"class_{class_id}", f"{iou:.6f}"])


def main():
    num_epochs = 50
    exp_name = experiment_name_with_epochs(
        "deeplabv3_resnet50_pretrained_road",
        num_epochs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = (
        PROJECT_ROOT / "outputs" / "checkpoints" / f"{exp_name}_best.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}\nRun python train_deeplab.py first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    protocol = checkpoint.get("label_protocol", {})
    num_classes = checkpoint.get(
        "num_classes",
        protocol.get("train_num_classes", CAMVID_12_ID_PROTOCOL.train_num_classes),
    )
    num_label_ids = protocol.get(
        "num_label_ids",
        CAMVID_12_ID_PROTOCOL.num_label_ids,
    )
    ignore_index = protocol.get(
        "ignore_index",
        CAMVID_12_ID_PROTOCOL.ignore_index,
    )
    image_size = tuple(checkpoint.get("image_size", (360, 480)))
    image_mean = tuple(checkpoint.get("image_mean", (0.485, 0.456, 0.406)))
    image_std = tuple(checkpoint.get("image_std", (0.229, 0.224, 0.225)))

    data_root = PROJECT_ROOT / "data" / "raw"
    eval_dataset = UAVSegmentationDataset(
        image_dir=data_root / "test" / "images",
        label_dir=data_root / "test" / "labels",
        image_size=image_size,
        augment=False,
        num_label_ids=num_label_ids,
        image_mean=image_mean,
        image_std=image_std,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_deeplabv3_resnet50(
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("=" * 80)
    print(f"Evaluating: {exp_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch')}  Val mIoU: {checkpoint.get('miou')}")
    print(f"Label protocol: {protocol or CAMVID_12_ID_PROTOCOL}")
    print(
        "Current eval split uses data/raw/test. Do not report it as formal "
        "CamVid test performance until the held-out test split is confirmed."
    )
    print("=" * 80)

    criterion = build_cross_entropy(ignore_index)
    metric = SegmentationMetric(num_classes=num_classes, ignore_index=ignore_index)
    eval_loss, results = evaluate_model(
        model=model,
        dataloader=eval_loader,
        criterion=criterion,
        device=device,
        metric=metric,
    )

    pixel_acc = results["Pixel Accuracy"]
    miou = results["Mean IoU"]
    class_iou = results["Class IoU"]
    class_ids = evaluated_class_ids(num_classes, ignore_index)

    metric_dir = PROJECT_ROOT / "outputs" / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    txt_save_path = metric_dir / f"{exp_name}_eval_results.txt"
    csv_save_path = metric_dir / f"{exp_name}_eval_results.csv"
    save_results_txt(
        txt_save_path,
        eval_loss,
        pixel_acc,
        miou,
        class_ids,
        class_iou,
    )
    save_results_csv(
        csv_save_path,
        eval_loss,
        pixel_acc,
        miou,
        class_ids,
        class_iou,
    )

    print(f"Eval Loss: {eval_loss:.6f}")
    print(f"Pixel Accuracy: {pixel_acc:.6f}")
    print(f"Mean IoU: {miou:.6f}")
    print(f"TXT results: {txt_save_path}")
    print(f"CSV results: {csv_save_path}")


if __name__ == "__main__":
    main()
