from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_color_map(num_classes=12):
    """
    为每个类别指定一种颜色。
    这里只是可视化用，不影响训练。
    """
    color_map = np.array([
        [0, 0, 0],         # class 0 - black
        [128, 0, 0],       # class 1 - maroon
        [0, 128, 0],       # class 2 - green
        [128, 128, 0],     # class 3 - olive
        [0, 0, 128],       # class 4 - navy
        [128, 0, 128],     # class 5 - purple
        [0, 128, 128],     # class 6 - teal
        [128, 128, 128],   # class 7 - gray
        [255, 0, 0],       # class 8 - red
        [0, 255, 0],       # class 9 - lime
        [0, 0, 255],       # class 10 - blue
        [255, 255, 0],     # class 11 - yellow
    ], dtype=np.uint8)

    if num_classes > len(color_map):
        raise ValueError(f"当前 color_map 只支持 {len(color_map)} 类，但你传入了 {num_classes} 类")

    return color_map[:num_classes]


def mask_to_color(mask, color_map):
    """
    将单通道类别编号图 [H, W]
    转换为彩色可视化图 [H, W, 3]
    """
    if mask.ndim != 2:
        raise ValueError(f"mask 应该是二维数组 [H, W]，但当前 shape = {mask.shape}")

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(color_map):
        color_mask[mask == class_id] = color

    return color_mask


def tensor_image_to_numpy(image_tensor):
    """
    将 PyTorch image tensor [3, H, W]
    转成 numpy uint8 图像 [H, W, 3]
    """
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))   # [H, W, 3]
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def overlay_mask_on_image(image, color_mask, alpha=0.5):
    """
    将彩色 mask 叠加到原图上。
    image: [H, W, 3], uint8
    color_mask: [H, W, 3], uint8
    """
    overlay = (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    return overlay


def save_comparison_figure(
    save_path,
    image,
    gt_color,
    pred_color,
    overlay_pred,
    filename=None
):
    """
    保存对比图：
    原图 / GT / Prediction / Overlay
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_color)
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(pred_color)
    axes[1, 0].set_title("Prediction")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay_pred)
    axes[1, 1].set_title("Prediction Overlay")
    axes[1, 1].axis("off")

    if filename is not None:
        fig.suptitle(filename, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_mask_image(save_path, color_mask):
    """
    单独保存彩色 mask 图
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(color_mask).save(save_path)