from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

# 把项目根目录加入 Python 搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.dataset import UAVSegmentationDataset


def main():
    data_root = PROJECT_ROOT / "data" / "raw"

    train_image_dir = data_root / "train" / "images"
    train_label_dir = data_root / "train" / "labels"

    dataset = UAVSegmentationDataset(
        image_dir=train_image_dir,
        label_dir=train_label_dir,
        image_size=(360, 480),
        augment=True,
    )

    print(f"训练集样本数量：{len(dataset)}")

    sample = dataset[0]

    image = sample["image"]
    label = sample["label"]
    filename = sample["filename"]

    print("=" * 80)
    print("单个样本检查")
    print("=" * 80)
    print(f"文件名：{filename}")
    print(f"image 类型：{type(image)}")
    print(f"label 类型：{type(label)}")
    print(f"image shape：{image.shape}")
    print(f"label shape：{label.shape}")
    print(f"image dtype：{image.dtype}")
    print(f"label dtype：{label.dtype}")
    print(f"image 最小值：{image.min().item():.4f}")
    print(f"image 最大值：{image.max().item():.4f}")
    print(f"label 类别编号：{torch.unique(label).tolist()}")

    print("=" * 80)
    print("DataLoader 检查")
    print("=" * 80)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(dataloader))

    images = batch["image"]
    labels = batch["label"]
    filenames = batch["filename"]

    print(f"batch images shape：{images.shape}")
    print(f"batch labels shape：{labels.shape}")
    print(f"batch filenames：{filenames}")

    print("=" * 80)
    print("Dataset 测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()