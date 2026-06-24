import os
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from models.vehicle_scene_relation_attention import add_vehicle_scene_relation_attention


# ==== 可选参数（按需手动修改） ====
CHECKPOINT_PATH = "outputs/segformer_camvid12_vsra/best_model.pth"
INPUT_PATH = "train/images"
OUTPUT_DIR = "outputs/predictions_camvid12_vsra_train"
NUM_CLASSES = 12
IGNORE_INDEX = 255
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENABLE_RELATION_ATTENTION = True
RELATION_CHANNELS = 64
RELATION_POOL_SIZE = 16
# 是否保存彩色分割图与叠加可视化
SAVE_COLOR = True
SAVE_OVERLAY = True
OVERLAY_ALPHA = 0.6

# Palette for classes (RGB tuples) — 可按需修改
PALETTE = [
    (128, 128, 128),  # Sky
    (128, 0, 0),      # Building
    (192, 192, 128),  # Pole
    (128, 64, 128),   # Road
    (60, 40, 222),    # Pavement
    (128, 128, 0),    # Tree
    (192, 128, 128),  # SignSymbol
    (64, 64, 128),    # Fence
    (64, 0, 128),     # Car
    (64, 64, 0),      # Pedestrian
    (0, 128, 192),    # Bicyclist
    (0, 0, 0),        # Unlabelled
]


def get_segformer_model() -> SegformerForSemanticSegmentation:
    config = SegformerConfig.from_pretrained(
        "nvidia/mit-b1",
        num_labels=NUM_CLASSES,
        local_files_only=True,
    )
    model = SegformerForSemanticSegmentation(config)
    model.config.num_labels = NUM_CLASSES
    model.config.semantic_loss_ignore_index = IGNORE_INDEX
    if ENABLE_RELATION_ATTENTION:
        model = add_vehicle_scene_relation_attention(
            model,
            relation_channels=RELATION_CHANNELS,
            pool_size=RELATION_POOL_SIZE,
        )
    return model


def list_images(path: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"输入路径不存在: {path}")
    files = []
    for name in os.listdir(path):
        if os.path.splitext(name)[1].lower() in exts:
            files.append(os.path.join(path, name))
    files.sort()
    return files


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image).unsqueeze(0)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        model.load_state_dict(state, strict=True)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(DEVICE)
    model: Any = get_segformer_model()
    if device.type == "cuda":
        model = model.cuda()
    else:
        model = model.cpu()
    load_checkpoint(model, CHECKPOINT_PATH)
    model.eval()

    image_paths = list_images(INPUT_PATH)
    if not image_paths:
        raise FileNotFoundError(f"未在目录中找到图像: {INPUT_PATH}")

    with torch.no_grad():
        for image_path in image_paths:
            rgb = Image.open(image_path).convert("RGB")
            rgb = np.array(rgb, dtype=np.uint8)
            h, w = rgb.shape[:2]

            tensor = preprocess_image(rgb).to(device)
            outputs = model(pixel_values=tensor)
            logits = outputs.logits
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            base = os.path.splitext(os.path.basename(image_path))[0]
            out_path = os.path.join(OUTPUT_DIR, f"{base}_mask.png")
            Image.fromarray(pred, mode="L").save(out_path)
            print(f"已保存灰度掩码: {out_path}")

            if SAVE_COLOR:
                # 将索引图映射为彩色 (使用 PALETTE 中的 RGB)
                palette_np = np.array(PALETTE, dtype=np.uint8)
                # pred: (H, W) -> color_mask: (H, W, 3) using palette indexing
                color_mask_rgb = palette_np[pred]
                out_color = os.path.join(OUTPUT_DIR, f"{base}_mask_color.png")
                Image.fromarray(color_mask_rgb, mode="RGB").save(out_color)
                print(f"已保存彩色掩码: {out_color}")

                if SAVE_OVERLAY:
                    # 原图与彩色 mask 做 alpha 叠加
                    overlay = (
                        rgb.astype(np.float32) * (1.0 - OVERLAY_ALPHA)
                        + color_mask_rgb.astype(np.float32) * OVERLAY_ALPHA
                    ).clip(0, 255).astype(np.uint8)
                    out_overlay = os.path.join(OUTPUT_DIR, f"{base}_overlay.png")
                    Image.fromarray(overlay, mode="RGB").save(out_overlay)
                    print(f"已保存叠加可视化: {out_overlay}")


if __name__ == "__main__":
    main()
