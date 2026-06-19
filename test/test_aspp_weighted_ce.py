"""测试 unet_aspp_weighted_ce 权重。"""

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.unet_aspp import UNetASPP
from test.test_aspp import save_results_csv, save_results_txt, test_model
from utils.dataset import UAVSegmentationDataset
from utils.experiment import experiment_name_with_epochs
from utils.losses import build_criterion


def main():
    num_epochs = 50
    exp_name = experiment_name_with_epochs("unet_aspp_weighted_ce", num_epochs)
    num_classes = 12
    image_size = (360, 480)
    batch_size = 2
    base_channels = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = PROJECT_ROOT / "data" / "raw"
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"{exp_name}_best.pth"
    output_metric_dir = PROJECT_ROOT / "outputs" / "metrics"
    output_metric_dir.mkdir(parents=True, exist_ok=True)
    txt_save_path = output_metric_dir / f"{exp_name}_test_results.txt"
    csv_save_path = output_metric_dir / f"{exp_name}_test_results.csv"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"缺少权重: {checkpoint_path}\n请先运行 python train_aspp_weighted_ce.py")

    test_loader = DataLoader(
        UAVSegmentationDataset(
            image_dir=data_root / "test" / "images",
            label_dir=data_root / "test" / "labels",
            image_size=image_size,
            augment=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = UNetASPP(3, num_classes, base_channels).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    class_weights = ckpt.get("class_weights")
    criterion = build_criterion(
        loss_name="weighted_ce",
        class_weights=class_weights,
        device=device,
    )

    print("=" * 80)
    print(f"Testing: {exp_name}")
    print(f"Epoch: {ckpt.get('epoch')}  Val mIoU: {ckpt.get('miou')}")
    print("=" * 80)

    test_loss, results = test_model(model, test_loader, criterion, device, num_classes)
    pixel_acc = results["Pixel Accuracy"]
    miou = results["Mean IoU"]
    class_iou = results["Class IoU"]
    print(f"Test mIoU: {miou:.6f}  Pixel Acc: {pixel_acc:.6f}")

    save_results_txt(txt_save_path, test_loss, pixel_acc, miou, class_iou)
    save_results_csv(csv_save_path, test_loss, pixel_acc, miou, class_iou)
    print(f"结果已保存: {txt_save_path}")


if __name__ == "__main__":
    main()

