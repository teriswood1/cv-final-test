import csv
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

# Config
NUM_CLASSES = 6
TRAIN_IMG_DIR = Path("train/images")
TRAIN_LABEL_DIR = Path("train/labels")
AUDIT_CSV = Path("outputs/label_audit/label_audit_files.csv")
OUT_DIR = Path("outputs/label_audit/visual_examples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# A simple extended palette
PALETTE = [
    (128, 64, 128),  # 0 Road
    (107, 142, 35),  # 1 Vegetation
    (0, 0, 255),     # 2 Water
    (70, 70, 70),    # 3 Building
    (220, 20, 60),   # 4 Vehicle
    (255, 165, 0),   # 5 Person
    (244, 35, 232),  # 6 extra
    (250, 170, 30),  # 7 extra
    (152, 251, 152), # 8 extra
    (153, 153, 153), # 9 extra
    (255, 255, 0),   # 10 extra
    (0, 255, 255),   # 11 extra
]


def load_candidates_from_audit() -> list:
    if AUDIT_CSV.exists():
        rows = []
        try:
            with open(AUDIT_CSV, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r['file'])
        except Exception:
            pass
        return rows
    return []


def scan_for_unexpected(limit=200):
    files = []
    for p in TRAIN_LABEL_DIR.glob("*.png"):
        try:
            a = np.array(Image.open(p))
            uniq = np.unique(a)
            if np.any(uniq >= NUM_CLASSES):
                files.append(p.name)
        except Exception:
            continue
        if len(files) >= limit:
            break
    return files


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    uniq = np.unique(mask)
    for v in uniq:
        if v == 255:
            color = (0, 0, 0)
        else:
            idx = int(v) if int(v) < len(PALETTE) else (int(v) % len(PALETTE))
            color = PALETTE[idx]
        out[mask == v] = color
    return out


def make_visuals(candidates: list, max_examples=10):
    chosen = random.sample(candidates, min(len(candidates), max_examples))
    summary = []
    for name in chosen:
        label_path = TRAIN_LABEL_DIR / name
        base = Path(name).stem
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
            cand = TRAIN_IMG_DIR / (base + ext)
            if cand.exists():
                img_path = cand
                break
        try:
            label = np.array(Image.open(label_path))
        except Exception:
            continue
        uniq = np.unique(label).tolist()
        summary.append((name, uniq))

        # load image if exists, else create a gray background
        if img_path and img_path.exists():
            img = np.array(Image.open(img_path).convert('RGB'))
        else:
            img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8) + 128

        color_label = colorize_mask(label)

        # overlay
        alpha = 0.6
        overlay = (img * (1 - alpha) + color_label * alpha).astype(np.uint8)

        Image.fromarray(img).save(OUT_DIR / f"{base}_orig.png")
        Image.fromarray(color_label).save(OUT_DIR / f"{base}_label_color.png")
        Image.fromarray(overlay).save(OUT_DIR / f"{base}_overlay.png")

    # write summary
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        for name, uniq in summary:
            f.write(f"{name}: {uniq}\n")


def main():
    candidates = load_candidates_from_audit()
    if not candidates:
        candidates = scan_for_unexpected(limit=500)
    if not candidates:
        print("未找到含异常标签的样例。")
        return
    make_visuals(candidates, max_examples=10)
    print(f"保存示例到 {OUT_DIR}")


if __name__ == '__main__':
    main()
