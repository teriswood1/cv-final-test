"""Detect possible illegal parking from SegFormer semantic masks.

Workflow:
1. Read semantic masks produced by `infer_segformer.py`.
2. Extract vehicle connected components.
3. Measure overlap with forbidden non-road classes.
4. Flag vehicles whose forbidden-overlap ratio is above threshold.
5. Save annotated images and a CSV summary.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import label, regionprops


PROJECT_ROOT = Path(__file__).resolve().parent

# 输入：语义分割输出目录（来自 infer_segformer.py）
MASK_DIR = PROJECT_ROOT / "outputs" / "predictions_train"

# 原图目录：用于画框与保存可视化
IMAGE_DIR = PROJECT_ROOT / "train" / "images"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "illegal_parking_train"

# 类别索引（与 infer_segformer.py 保持一致）
ROAD_CLASS = 0
VEHICLE_CLASS = 4
PERSON_CLASS = 5
VEGETATION_CLASS = 1
WATER_CLASS = 2
BUILDING_CLASS = 3

# 认为是“非道路禁停区域”的类别，可按需求调整
FORBIDDEN_CLASSES = {VEGETATION_CLASS, WATER_CLASS, BUILDING_CLASS, PERSON_CLASS}

# 过滤小噪声连通域
MIN_VEHICLE_AREA = 80

# 判定阈值：车辆外扩上下文中，被禁停区域覆盖的比例
FORBIDDEN_CONTEXT_THRESHOLD = 0.35

# 判定阈值：车辆外扩上下文中，道路像素占比过低时更可疑
ROAD_CONTEXT_THRESHOLD = 0.25

# 车辆外接框外扩像素，越大越看“周边环境”
CONTEXT_MARGIN = 18

# 画框颜色（BGR）
FLAG_COLOR = (0, 0, 255)
NORMAL_COLOR = (0, 255, 0)


@dataclass
class ParkingResult:
    image_name: str
    component_id: int
    x: int
    y: int
    w: int
    h: int
    area: int
    road_context_ratio: float
    forbidden_context_ratio: float
    vehicle_coverage: float
    is_illegal: bool


def list_mask_files(mask_dir: Path) -> List[Path]:
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    files = sorted(mask_dir.glob("*_mask.png"))
    if not files:
        raise FileNotFoundError(f"No *_mask.png files found in: {mask_dir}")
    return files


def load_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    return np.array(mask, dtype=np.uint8)


def load_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.array(image, dtype=np.uint8)


def connected_components(vehicle_mask: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    labels = label(vehicle_mask.astype(np.uint8), connectivity=2)
    props = regionprops(labels)
    return len(props) + 1, labels, props, None


def crop_with_margin(
    mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    margin: int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    height, width = mask.shape[:2]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(width, x + w + margin)
    y1 = min(height, y + h + margin)
    return mask[y0:y1, x0:x1], (x0, y0, x1, y1)


def infer_image_path_from_mask(mask_path: Path) -> Path:
    base = mask_path.name.replace("_mask.png", "")
    candidates = [
        IMAGE_DIR / f"{base}.png",
        IMAGE_DIR / f"{base}.jpg",
        IMAGE_DIR / f"{base}.jpeg",
        IMAGE_DIR / f"{base}.bmp",
        IMAGE_DIR / f"{base}.tif",
        IMAGE_DIR / f"{base}.tiff",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find original image for mask: {mask_path}")


def draw_result_box(
    image: Image.Image,
    box: Tuple[int, int, int, int],
    text: str,
    color: Tuple[int, int, int],
) -> Image.Image:
    x, y, w, h = box
    draw = ImageDraw.Draw(image)
    draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
    label_y = max(0, y - 14)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    # 画一个小背景块，保证文字可见
    text_bbox = draw.textbbox((x, label_y), text, font=font)
    pad = 2
    draw.rectangle(
        [text_bbox[0] - pad, text_bbox[1] - pad, text_bbox[2] + pad, text_bbox[3] + pad],
        fill=(0, 0, 0),
    )
    draw.text((x, label_y), text, fill=color, font=font)
    return image


def detect_from_mask(mask_path: Path) -> Tuple[List[ParkingResult], np.ndarray]:
    mask = load_mask(mask_path)
    vehicle_mask = mask == VEHICLE_CLASS

    num_labels, labels, props, _ = connected_components(vehicle_mask)
    results: List[ParkingResult] = []

    for component_id in range(1, num_labels):
        component_prop = props[component_id - 1]
        min_row, min_col, max_row, max_col = component_prop.bbox
        x = int(min_col)
        y = int(min_row)
        w = int(max_col - min_col)
        h = int(max_row - min_row)
        area = int(component_prop.area)

        if area < MIN_VEHICLE_AREA:
            continue

        component_mask = labels == component_id
        component_area = int(component_mask.sum())

        # 取车辆外接框的外扩上下文，而不是只看车辆像素本身
        context_crop, (x0, y0, x1, y1) = crop_with_margin(
            mask, x, y, w, h, CONTEXT_MARGIN
        )
        crop_vehicle = component_mask[y0:y1, x0:x1]

        context_area = int(context_crop.size - crop_vehicle.sum())
        if context_area <= 0:
            continue

        context_without_vehicle = context_crop[~crop_vehicle]
        road_pixels = int((context_without_vehicle == ROAD_CLASS).sum())
        forbidden_pixels = int(np.isin(context_without_vehicle, list(FORBIDDEN_CLASSES)).sum())

        road_context_ratio = road_pixels / context_area
        forbidden_context_ratio = forbidden_pixels / context_area
        vehicle_coverage = component_area / float((x1 - x0) * (y1 - y0))

        is_illegal = (
            forbidden_context_ratio >= FORBIDDEN_CONTEXT_THRESHOLD
            and road_context_ratio <= ROAD_CONTEXT_THRESHOLD
        )

        results.append(
            ParkingResult(
                image_name=mask_path.name.replace("_mask.png", ""),
                component_id=component_id,
                x=x,
                y=y,
                w=w,
                h=h,
                area=component_area,
                road_context_ratio=road_context_ratio,
                forbidden_context_ratio=forbidden_context_ratio,
                vehicle_coverage=vehicle_coverage,
                is_illegal=is_illegal,
            )
        )

    return results, mask


def save_csv(results: Sequence[ParkingResult], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_name",
                "component_id",
                "x",
                "y",
                "w",
                "h",
                "area",
                "road_context_ratio",
                "forbidden_context_ratio",
                "vehicle_coverage",
                "is_illegal",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.image_name,
                    item.component_id,
                    item.x,
                    item.y,
                    item.w,
                    item.h,
                    item.area,
                    f"{item.road_context_ratio:.6f}",
                    f"{item.forbidden_context_ratio:.6f}",
                    f"{item.vehicle_coverage:.6f}",
                    int(item.is_illegal),
                ]
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_files = list_mask_files(MASK_DIR)

    all_results: List[ParkingResult] = []
    summary_lines: List[str] = []

    for mask_path in mask_files:
        image_path = infer_image_path_from_mask(mask_path)
        image = Image.fromarray(load_image(image_path))
        results, _ = detect_from_mask(mask_path)

        illegal_count = 0
        for item in results:
            all_results.append(item)
            box = (item.x, item.y, item.w, item.h)
            if item.is_illegal:
                illegal_count += 1
                text = f"ILLEGAL {item.forbidden_context_ratio:.2f}"
                image = draw_result_box(image, box, text, FLAG_COLOR)
            else:
                text = f"OK {item.forbidden_context_ratio:.2f}"
                image = draw_result_box(image, box, text, NORMAL_COLOR)

        out_image = OUTPUT_DIR / f"{mask_path.name.replace('_mask.png', '')}_illegal_parking.png"
        image.save(out_image)

        summary_lines.append(
            f"{mask_path.name}: vehicles={len(results)}, illegal={illegal_count}, saved={out_image.name}"
        )
        print(summary_lines[-1])

    csv_path = OUTPUT_DIR / "illegal_parking_results.csv"
    save_csv(all_results, csv_path)

    txt_path = OUTPUT_DIR / "illegal_parking_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Illegal parking detection summary\n")
        f.write("=" * 60 + "\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {txt_path}")


if __name__ == "__main__":
    main()
