"""Audit training segmentation labels.

Checks:
- Every label file in `train/labels` has a matching image in `train/images`.
- Image and label sizes match.
- Reports unique label values and unexpected values (outside 0..NUM_CLASSES-1 and 255).
- Aggregates per-class pixel counts and image presence.

Outputs:
- outputs/label_audit/label_audit_summary.txt
- outputs/label_audit/label_audit_files.csv
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_ROOT / "train" / "images"
LABEL_DIR = PROJECT_ROOT / "train" / "labels"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "label_audit"

CLASS_NAMES = ["Road", "Vegetation", "Water", "Building", "Vehicle", "Person"]
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255


def list_files(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.suffix.lower() in exts])


def find_matching_image(label_path: Path) -> Path | None:
    stem = label_path.stem
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        candidate = IMAGE_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def load_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size[1], im.size[0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_files = list_files(LABEL_DIR)
    if not label_files:
        print(f"No label files found in {LABEL_DIR}")
        return

    missing_images = []
    size_mismatches: List[str] = []
    unexpected_values: List[str] = []
    per_class_pixels = Counter()
    per_class_images = Counter()
    file_rows = []

    total_pixels = 0

    for label_path in label_files:
        image_path = find_matching_image(label_path)
        if image_path is None:
            missing_images.append(label_path.name)
            continue

        label = load_mask(label_path)
        lh, lw = label.shape[:2]
        ih, iw = load_image_size(image_path)

        if (lh, lw) != (ih, iw):
            size_mismatches.append(f"{label_path.name}: label={lw}x{lh} image={iw}x{ih}")

        unique = np.unique(label)
        unexpected = [int(v) for v in unique if int(v) not in range(NUM_CLASSES) and int(v) != IGNORE_INDEX]
        if unexpected:
            unexpected_values.append(f"{label_path.name}: {unexpected}")

        total_pixels += label.size
        for v in range(NUM_CLASSES):
            cnt = int((label == v).sum())
            if cnt > 0:
                per_class_pixels[v] += cnt
                per_class_images[v] += 1

        file_rows.append({
            "file": label_path.name,
            "unique_values": ",".join(map(str, unique.tolist())),
            "unexpected_values": ",".join(map(str, unexpected)) if unexpected else "",
            "size_ok": (lh, lw) == (ih, iw),
        })

    summary_path = OUTPUT_DIR / "label_audit_summary.txt"
    csv_path = OUTPUT_DIR / "label_audit_files.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Label audit summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Label files scanned: {len(label_files)}\n")
        f.write(f"Missing matched images: {len(missing_images)}\n")
        f.write(f"Size mismatches: {len(size_mismatches)}\n")
        f.write(f"Files with unexpected values: {len(unexpected_values)}\n\n")

        f.write("Per-class pixel counts:\n")
        for i, name in enumerate(CLASS_NAMES):
            cnt = per_class_pixels[i]
            pct = cnt / total_pixels if total_pixels > 0 else 0.0
            f.write(f"  {i} {name}: {cnt} pixels ({pct:.4%})\n")

        if missing_images:
            f.write("\nMissing images (examples):\n")
            for item in missing_images[:50]:
                f.write(f"  {item}\n")

        if size_mismatches:
            f.write("\nSize mismatches (examples):\n")
            for item in size_mismatches[:50]:
                f.write(f"  {item}\n")

        if unexpected_values:
            f.write("\nUnexpected label values (examples):\n")
            for item in unexpected_values[:50]:
                f.write(f"  {item}\n")

    import csv

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "unique_values", "unexpected_values", "size_ok"])
        writer.writeheader()
        for r in file_rows:
            writer.writerow(r)

    print("Label audit complete")
    print(f"Summary: {summary_path}")
    print(f"Per-file CSV: {csv_path}")


if __name__ == "__main__":
    main()
