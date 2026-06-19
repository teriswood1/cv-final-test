from pathlib import Path
import sys
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.dataset import UAVSegmentationDataset
from utils.experiment import experiment_name_with_epochs
from utils.metrics import SegmentationMetric
from models.unet_cbam_aspp import UNetCBAMASPP


@torch.no_grad()
def test_model(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    metric = SegmentationMetric(num_classes=num_classes)

    for batch in tqdm(dataloader, desc="Testing"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        metric.update(outputs, labels)

    return total_loss / len(dataloader), metric.get_results()


def save_results_txt(save_path, test_loss, pixel_acc, miou, class_iou):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("U-Net + CBAM + ASPP Test Results\n")
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
    num_epochs = 50
    exp_name = experiment_name_with_epochs("unet_cbam_aspp_augfix", num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("U-Net + CBAM + ASPP Testing")
    print("=" * 80)
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"使用设备：{device}")
    print("=" * 80)

    data_root = PROJECT_ROOT / "data" / "raw"
    test_image_dir = data_root / "test" / "images"
    test_label_dir = data_root / "test" / "labels"

    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"{exp_name}_best.pth"
    output_metric_dir = PROJECT_ROOT / "outputs" / "metrics"
    output_metric_dir.mkdir(parents=True, exist_ok=True)

    txt_save_path = output_metric_dir / f"{exp_name}_test_results.txt"
    csv_save_path = output_metric_dir / f"{exp_name}_test_results.csv"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到模型权重文件：{checkpoint_path}")

    test_dataset = UAVSegmentationDataset(
        image_dir=test_image_dir,
        label_dir=test_label_dir,
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

    print(f"测试集数量：{len(test_dataset)}")

    model = UNetCBAMASPP(
        in_channels=3,
        num_classes=num_classes,
        base_channels=base_channels,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"已加载模型：{checkpoint_path}")
    print(f"模型来自 Epoch：{checkpoint.get('epoch', 'unknown')}")
    print(f"验证集 mIoU：{checkpoint.get('miou', 'unknown')}")

    test_loss, results = test_model(
        model, test_loader, nn.CrossEntropyLoss(), device, num_classes
    )

    pixel_acc = results["Pixel Accuracy"]
    miou = results["Mean IoU"]
    class_iou = results["Class IoU"]

    print("=" * 80)
    print(f"Test Loss:      {test_loss:.6f}")
    print(f"Pixel Accuracy: {pixel_acc:.6f}")
    print(f"Mean IoU:       {miou:.6f}")
    for i, iou in enumerate(class_iou):
        print(f"class_{i}: {iou:.6f}")

    save_results_txt(txt_save_path, test_loss, pixel_acc, miou, class_iou)
    save_results_csv(csv_save_path, test_loss, pixel_acc, miou, class_iou)

    print("=" * 80)
    print(f"TXT 结果：{txt_save_path}")
    print(f"CSV 结果：{csv_save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
