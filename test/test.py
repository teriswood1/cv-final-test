from pathlib import Path
import sys
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.dataset import UAVSegmentationDataset
from utils.metrics import SegmentationMetric
from models.unet import UNet


@torch.no_grad()
def test_model(model, dataloader, criterion, device, num_classes):
    """
    在测试集上评估模型。
    """
    model.eval()

    total_loss = 0.0
    metric = SegmentationMetric(num_classes=num_classes)

    progress_bar = tqdm(dataloader, desc="Testing")

    for batch in progress_bar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        metric.update(outputs, labels)

    avg_loss = total_loss / len(dataloader)
    results = metric.get_results()

    return avg_loss, results


def save_results_txt(save_path, test_loss, pixel_acc, miou, class_iou):
    """
    保存 txt 格式测试结果。
    """
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("U-Net Baseline Test Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Pixel Accuracy: {pixel_acc:.6f}\n")
        f.write(f"Mean IoU: {miou:.6f}\n")
        f.write("\nClass IoU:\n")

        for i, iou in enumerate(class_iou):
            f.write(f"class_{i}: {iou:.6f}\n")


def save_results_csv(save_path, test_loss, pixel_acc, miou, class_iou):
    """
    保存 csv 格式测试结果，方便后面做表格。
    """
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
    # =========================
    # 1. 基础配置
    # =========================
    num_classes = 12
    image_size = (360, 480)
    batch_size = 2
    base_channels = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("U-Net Baseline Testing")
    print("=" * 80)
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"使用设备：{device}")
    print(f"类别数量：{num_classes}")
    print(f"输入尺寸：{image_size}")
    print("=" * 80)

    # =========================
    # 2. 路径设置
    # =========================
    data_root = PROJECT_ROOT / "data" / "raw"

    test_image_dir = data_root / "test" / "images"
    test_label_dir = data_root / "test" / "labels"

    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "unet_baseline_best.pth"

    output_metric_dir = PROJECT_ROOT / "outputs" / "metrics"
    output_metric_dir.mkdir(parents=True, exist_ok=True)

    txt_save_path = output_metric_dir / "unet_baseline_test_results.txt"
    csv_save_path = output_metric_dir / "unet_baseline_test_results.csv"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到模型权重文件：{checkpoint_path}")

    # =========================
    # 3. 构建测试集
    # =========================
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
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"测试集数量：{len(test_dataset)}")

    # =========================
    # 4. 构建模型并加载权重
    # =========================
    model = UNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=base_channels,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"已加载模型：{checkpoint_path}")
    print(f"模型来自 Epoch：{checkpoint.get('epoch', 'unknown')}")
    print(f"验证集 mIoU：{checkpoint.get('miou', 'unknown')}")
    print(f"验证集 Pixel Acc：{checkpoint.get('pixel_acc', 'unknown')}")

    criterion = nn.CrossEntropyLoss()

    # =========================
    # 5. 测试
    # =========================
    test_loss, results = test_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
    )

    pixel_acc = results["Pixel Accuracy"]
    miou = results["Mean IoU"]
    class_iou = results["Class IoU"]

    print("=" * 80)
    print("测试结果")
    print("=" * 80)
    print(f"Test Loss:      {test_loss:.6f}")
    print(f"Pixel Accuracy: {pixel_acc:.6f}")
    print(f"Mean IoU:       {miou:.6f}")
    print("-" * 80)
    print("Class IoU:")

    for i, iou in enumerate(class_iou):
        print(f"class_{i}: {iou:.6f}")

    # =========================
    # 6. 保存结果
    # =========================
    save_results_txt(
        save_path=txt_save_path,
        test_loss=test_loss,
        pixel_acc=pixel_acc,
        miou=miou,
        class_iou=class_iou,
    )

    save_results_csv(
        save_path=csv_save_path,
        test_loss=test_loss,
        pixel_acc=pixel_acc,
        miou=miou,
        class_iou=class_iou,
    )

    print("=" * 80)
    print("测试完成")
    print(f"TXT 结果保存到：{txt_save_path}")
    print(f"CSV 结果保存到：{csv_save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
