from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from utils.dataset import UAVSegmentationDataset


def write_pair(root: Path, label_values, filename="sample.png"):
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    label_array = np.array(label_values, dtype=np.uint8)
    image_array = np.zeros((*label_array.shape, 3), dtype=np.uint8)

    Image.fromarray(image_array, mode="RGB").save(image_dir / filename)
    Image.fromarray(label_array, mode="L").save(label_dir / filename)

    return image_dir, label_dir


def test_dataset_rejects_unconfigured_label_id(tmp_path):
    image_dir, label_dir = write_pair(tmp_path, [[0, 1, 2, 3]] * 4)
    dataset = UAVSegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=(4, 4),
        num_label_ids=3,
    )

    with pytest.raises(ValueError, match="outside configured label ids"):
        dataset[0]


def test_road_augmentation_preserves_mask_ids(tmp_path, monkeypatch):
    image_dir, label_dir = write_pair(tmp_path, [[0, 1, 1, 0]] * 4)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([1.0]))
    dataset = UAVSegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=(4, 4),
        augment=True,
        augment_mode="road",
        num_label_ids=2,
    )

    sample = dataset[0]

    assert set(sample["label"].unique().tolist()) <= {0, 1}


def test_dataset_applies_configured_image_normalization(tmp_path):
    image_dir, label_dir = write_pair(tmp_path, [[0, 1], [1, 0]])
    dataset = UAVSegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=(2, 2),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )

    sample = dataset[0]

    assert sample["image"].min().item() == -1.0
    assert sample["image"].max().item() == -1.0
