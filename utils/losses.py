"""
分割损失：标准 CE、类别加权 CE。
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from utils.dataset import UAVSegmentationDataset


def compute_class_pixel_counts(
    label_dir,
    image_dir,
    indices,
    num_classes=12,
):
    """统计指定索引样本中每类像素数。"""
    label_dir = Path(label_dir)
    image_dir = Path(image_dir)

    dataset = UAVSegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=(360, 480),
        augment=False,
    )

    counts = np.zeros(num_classes, dtype=np.int64)

    for idx in tqdm(indices, desc="统计类别像素", leave=False):
        sample = dataset[idx]
        label = sample["label"].numpy()
        for c in range(num_classes):
            counts[c] += int((label == c).sum())

    return counts


def counts_to_weights(
    counts,
    num_classes=12,
    mode="median_freq",
    clip_max=None,
):
    """
    由像素计数得到类别权重。

    mode:
        - median_freq:  median / (freq + eps)，再归一化使均值为 1
        - inverse_sqrt: 1 / sqrt(freq)
    """
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)

    if mode == "median_freq":
        freq = counts / counts.sum()
        median = np.median(freq[freq > 0])
        weights = median / (freq + 1e-8)
    elif mode == "inverse_sqrt":
        freq = counts / counts.sum()
        weights = 1.0 / np.sqrt(freq + 1e-8)
    else:
        raise ValueError(f"未知 mode: {mode}")

    weights = weights / weights.mean()
    if clip_max is not None:
        weights = np.clip(weights, 0.0, float(clip_max))
        weights = weights / max(weights.mean(), 1e-8)
    return torch.tensor(weights, dtype=torch.float32)


def build_criterion(
    loss_name="ce",
    class_weights=None,
    num_classes=12,
    device="cpu",
):
    """
    loss_name: 'ce' | 'weighted_ce'
    """
    if loss_name == "ce":
        return nn.CrossEntropyLoss()

    if loss_name == "weighted_ce":
        if class_weights is None:
            raise ValueError("weighted_ce 需要提供 class_weights")
        weight = class_weights.to(device)
        return nn.CrossEntropyLoss(weight=weight)

    raise ValueError(f"未知 loss: {loss_name}")
