from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.dataset import UAVSegmentationDataset
from utils.visualize import (
    get_color_map,
    mask_to_color,
    tensor_image_to_numpy,
    overlay_mask_on_image,
    save_comparison_figure,
    save_mask_image,
)
from models.unet import UNet


@torch.no_grad()
def run_prediction(model, dataloader, device, color_map, save_dir, max_save=None):
    """
    对测试集进行预测并保存可视化结果。

    max_save:
        None 表示全部保存
        整数 n 表示只保存前 n 张
    """
    model.eval()

    compare_dir = save_dir / "compare"
    pred_mask_dir = save_dir / "pred_masks"
    gt_mask_dir = save_dir / "gt_masks"

    compare_dir.mkdir(parents=True, exist_ok=True)
    pred_mask_dir.mkdir(parents=True, exist_ok=True)
    gt_mask_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    progress_bar = tqdm(dataloader, desc="Predicting")

    for batch in progress_bar:
        images = batch["image"].to(device)      # [B, 3, H, W]
        labels = batch["label"]                 # [B, H, W]
        filenames = batch["filename"]

        outputs = model(images)                 # [B, C, H, W]
        preds = torch.argmax(outputs, dim=1).cpu()   # [B, H, W]

        for i in range(images.size(0)):
            if max_save is not None and saved_count >= max_save:
                return

            filename = filenames[i]

            # 原图
            image_np = tensor_image_to_numpy(images[i].cpu())

            # GT / Pred
            gt_mask = labels[i].cpu().numpy()
            pred_mask = preds[i].numpy()

            # 彩色图
            gt_color = mask_to_color(gt_mask, color_map)
            pred_color = mask_to_color(pred_mask, color_map)

            # overlay
            overlay_pred = overlay_mask_on_image(image_np, pred_color, alpha=0.5)

            # 文件名
            stem = Path(filename).stem

            compare_path = compare_dir / f"{stem}_compare.png"
            pred_mask_path = pred_mask_dir / f"{stem}_pred.png"
            gt_mask_path = gt_mask_dir / f"{stem}_gt.png"

            # 保存
            save_comparison_figure(
                save_path=compare_path,
                image=image_np,
                gt_color=gt_color,
                pred_color=pred_color,
                overlay_pred=overlay_pred,
                filename=filename,
            )

            save_mask_image(pred_mask_path, pred_color)
            save_mask_image(gt_mask_path, gt_color)

            saved_count += 1


def main():
    # =========================
    # 1. 基础配置
    # =========================
    num_classes = 12
    image_size = (360, 480)
    batch_size = 1
    base_channels = 32

    # 保存多少张可视化图：
    # None = 全部保存
    # 比如 20 = 只保存前 20 张
    max_save = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("U-Net Baseline Prediction")
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

    save_dir = PROJECT_ROOT / "outputs" / "predictions" / "unet_baseline"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到模型权重文件：{checkpoint_path}")

    # =========================
    # 3. 数据集与 DataLoader
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
    # 4. 加载模型
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

    # =========================
    # 5. 可视化预测
    # =========================
    color_map = get_color_map(num_classes=num_classes)

    run_prediction(
        model=model,
        dataloader=test_loader,
        device=device,
        color_map=color_map,
        save_dir=save_dir,
        max_save=max_save,
    )

    print("=" * 80)
    print("预测完成")
    print(f"可视化结果保存到：{save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
