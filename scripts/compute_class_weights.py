"""
根据训练集（划分后的 train 索引）统计类别权重并保存。
"""

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.data_split import create_train_val_datasets
from utils.losses import compute_class_pixel_counts, counts_to_weights


def main():
    num_classes = 12
    val_ratio = 0.2
    random_seed = 42

    data_root = PROJECT_ROOT / "data" / "raw"
    image_dir = data_root / "train" / "images"
    label_dir = data_root / "train" / "labels"

    train_dataset, _, train_indices, _ = create_train_val_datasets(
        image_dir=image_dir,
        label_dir=label_dir,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )

    print(f"训练样本数：{len(train_indices)}")
    print(f"训练集 augment={train_dataset.dataset.augment}")

    counts = compute_class_pixel_counts(
        label_dir=label_dir,
        image_dir=image_dir,
        indices=train_indices,
        num_classes=num_classes,
    )

    weights = counts_to_weights(counts, num_classes=num_classes)

    save_dir = PROJECT_ROOT / "outputs"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "class_weights.pt"

    torch.save(
        {
            "counts": counts,
            "weights": weights,
            "num_classes": num_classes,
            "val_ratio": val_ratio,
            "random_seed": random_seed,
        },
        save_path,
    )

    print("=" * 60)
    print("类别像素统计与权重")
    for i in range(num_classes):
        print(
            f"class_{i}: pixels={counts[i]:>12,}  weight={weights[i].item():.4f}"
        )
    print("=" * 60)
    print(f"已保存到：{save_path}")


if __name__ == "__main__":
    main()
