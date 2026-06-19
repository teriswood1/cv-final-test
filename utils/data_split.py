"""
训练 / 验证集划分工具。

修复 random_split 与单一 Dataset 共享 augment 标志的问题：
训练集、验证集使用独立的 Dataset 实例，再对各自做 Subset。
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from utils.dataset import UAVSegmentationDataset


def create_train_val_datasets(
    image_dir,
    label_dir,
    image_size=(360, 480),
    val_ratio=0.2,
    random_seed=42,
    train_augment_mode="strong",
    num_label_ids=None,
    image_mean=None,
    image_std=None,
):
    """
    返回 (train_dataset, val_dataset, train_indices, val_indices)。

    - 训练集：augment=True
    - 验证集：augment=False
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    probe = UAVSegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=image_size,
        augment=False,
        num_label_ids=num_label_ids,
        image_mean=image_mean,
        image_std=image_std,
    )
    n = len(probe)
    del probe

    val_size = int(n * val_ratio)
    train_size = n - val_size

    generator = torch.Generator().manual_seed(random_seed)
    indices = torch.randperm(n, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_full = UAVSegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=image_size,
        augment=True,
        augment_mode=train_augment_mode,
        num_label_ids=num_label_ids,
        image_mean=image_mean,
        image_std=image_std,
    )
    val_full = UAVSegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=image_size,
        augment=False,
        num_label_ids=num_label_ids,
        image_mean=image_mean,
        image_std=image_std,
    )

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    return train_dataset, val_dataset, train_indices, val_indices


def create_train_val_loaders(
    image_dir,
    label_dir,
    image_size=(360, 480),
    batch_size=2,
    val_ratio=0.2,
    random_seed=42,
    device_type="cpu",
    train_augment_mode="strong",
    num_label_ids=None,
    image_mean=None,
    image_std=None,
):
    """构建 DataLoader，并返回划分信息。"""
    train_dataset, val_dataset, train_indices, val_indices = create_train_val_datasets(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=image_size,
        val_ratio=val_ratio,
        random_seed=random_seed,
        train_augment_mode=train_augment_mode,
        num_label_ids=num_label_ids,
        image_mean=image_mean,
        image_std=image_std,
    )

    pin_memory = device_type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    info = {
        "total": len(train_indices) + len(val_indices),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "train_augment": train_dataset.dataset.augment,
        "train_augment_mode": train_dataset.dataset.augment_mode,
        "val_augment": val_dataset.dataset.augment,
    }

    return train_loader, val_loader, info


def create_train_eval_loaders(
    train_image_dir,
    train_label_dir,
    eval_image_dir,
    eval_label_dir,
    image_size=(360, 480),
    batch_size=2,
    device_type="cpu",
    train_augment_mode="road",
    num_label_ids=None,
    image_mean=None,
    image_std=None,
    train_drop_last=True,
):
    """构建完整训练目录与独立评估目录的 DataLoader。"""
    train_dataset = UAVSegmentationDataset(
        image_dir=train_image_dir,
        label_dir=train_label_dir,
        image_size=image_size,
        augment=True,
        augment_mode=train_augment_mode,
        num_label_ids=num_label_ids,
        image_mean=image_mean,
        image_std=image_std,
    )
    eval_dataset = UAVSegmentationDataset(
        image_dir=eval_image_dir,
        label_dir=eval_label_dir,
        image_size=image_size,
        augment=False,
        num_label_ids=num_label_ids,
        image_mean=image_mean,
        image_std=image_std,
    )

    pin_memory = device_type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=train_drop_last,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    info = {
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "train_augment": train_dataset.augment,
        "train_augment_mode": train_dataset.augment_mode,
        "train_drop_last": train_drop_last,
        "eval_augment": eval_dataset.augment,
    }
    return train_loader, eval_loader, info


def verify_augment_flags(train_dataset, val_dataset):
    """确认训练集开启增强、验证集关闭增强。"""
    train_aug = train_dataset.dataset.augment
    val_aug = val_dataset.dataset.augment
    if not train_aug or val_aug:
        raise RuntimeError(
            f"数据增强配置异常: train_augment={train_aug}, val_augment={val_aug} "
            f"(期望 True, False)"
        )
    return train_aug, val_aug
