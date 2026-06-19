from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from models.unet_cbam_aspp import UNetCBAMASPP


@torch.no_grad()
def run_prediction(model, dataloader, device, color_map, save_dir, max_save=None):
    model.eval()

    compare_dir = save_dir / "compare"
    pred_mask_dir = save_dir / "pred_masks"
    gt_mask_dir = save_dir / "gt_masks"
    compare_dir.mkdir(parents=True, exist_ok=True)
    pred_mask_dir.mkdir(parents=True, exist_ok=True)
    gt_mask_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for batch in tqdm(dataloader, desc="Predicting"):
        images = batch["image"].to(device)
        labels = batch["label"]
        filenames = batch["filename"]

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu()

        for i in range(images.size(0)):
            if max_save is not None and saved_count >= max_save:
                return

            filename = filenames[i]
            image_np = tensor_image_to_numpy(images[i].cpu())
            gt_mask = labels[i].cpu().numpy()
            pred_mask = preds[i].numpy()

            gt_color = mask_to_color(gt_mask, color_map)
            pred_color = mask_to_color(pred_mask, color_map)
            overlay_pred = overlay_mask_on_image(image_np, pred_color, alpha=0.5)

            stem = Path(filename).stem
            save_comparison_figure(
                save_path=compare_dir / f"{stem}_compare.png",
                image=image_np,
                gt_color=gt_color,
                pred_color=pred_color,
                overlay_pred=overlay_pred,
                filename=filename,
            )
            save_mask_image(pred_mask_dir / f"{stem}_pred.png", pred_color)
            save_mask_image(gt_mask_dir / f"{stem}_gt.png", gt_color)
            saved_count += 1


def main():
    num_classes = 12
    image_size = (360, 480)
    batch_size = 1
    base_channels = 32
    max_save = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("U-Net + CBAM + ASPP Prediction")
    print("=" * 80)

    data_root = PROJECT_ROOT / "data" / "raw"
    test_image_dir = data_root / "test" / "images"
    test_label_dir = data_root / "test" / "labels"

    checkpoint_path = (
        PROJECT_ROOT / "outputs" / "checkpoints" / "unet_cbam_aspp_best.pth"
    )
    save_dir = PROJECT_ROOT / "outputs" / "predictions" / "unet_cbam_aspp"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到模型权重文件：{checkpoint_path}")

    test_loader = DataLoader(
        UAVSegmentationDataset(
            image_dir=test_image_dir,
            label_dir=test_label_dir,
            image_size=image_size,
            augment=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = UNetCBAMASPP(
        in_channels=3,
        num_classes=num_classes,
        base_channels=base_channels,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"已加载模型：{checkpoint_path}")

    run_prediction(
        model=model,
        dataloader=test_loader,
        device=device,
        color_map=get_color_map(num_classes=num_classes),
        save_dir=save_dir,
        max_save=max_save,
    )

    print("=" * 80)
    print(f"可视化结果保存到：{save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
