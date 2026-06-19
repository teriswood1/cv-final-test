from pathlib import Path

import numpy as np
from PIL import Image

from utils.data_split import create_train_eval_loaders


def write_pair(root: Path, filename: str, label_values):
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    label_array = np.array(label_values, dtype=np.uint8)
    image_array = np.zeros((*label_array.shape, 3), dtype=np.uint8)
    Image.fromarray(image_array, mode="RGB").save(image_dir / filename)
    Image.fromarray(label_array, mode="L").save(label_dir / filename)

    return image_dir, label_dir


def test_train_eval_loaders_do_not_resplit_training_directory(tmp_path):
    train_images, train_labels = write_pair(
        tmp_path / "train",
        "sample_1.png",
        [[0, 1], [1, 0]],
    )
    write_pair(tmp_path / "train", "sample_2.png", [[1, 0], [0, 1]])
    eval_images, eval_labels = write_pair(
        tmp_path / "eval",
        "sample_eval.png",
        [[0, 1], [1, 0]],
    )

    train_loader, eval_loader, info = create_train_eval_loaders(
        train_image_dir=train_images,
        train_label_dir=train_labels,
        eval_image_dir=eval_images,
        eval_label_dir=eval_labels,
        image_size=(2, 2),
        batch_size=1,
        device_type="cpu",
        num_label_ids=2,
        train_augment_mode="road",
    )

    assert len(train_loader.dataset) == 2
    assert len(eval_loader.dataset) == 1
    assert info["train_size"] == 2
    assert info["eval_size"] == 1


def test_train_eval_loader_drops_incomplete_training_batch(tmp_path):
    train_images, train_labels = write_pair(
        tmp_path / "train",
        "sample_1.png",
        [[0, 1], [1, 0]],
    )
    write_pair(tmp_path / "train", "sample_2.png", [[1, 0], [0, 1]])
    write_pair(tmp_path / "train", "sample_3.png", [[0, 0], [1, 1]])
    eval_images, eval_labels = write_pair(
        tmp_path / "eval",
        "sample_eval.png",
        [[0, 1], [1, 0]],
    )

    train_loader, _, info = create_train_eval_loaders(
        train_image_dir=train_images,
        train_label_dir=train_labels,
        eval_image_dir=eval_images,
        eval_label_dir=eval_labels,
        image_size=(2, 2),
        batch_size=2,
        device_type="cpu",
        num_label_ids=2,
        train_augment_mode="road",
    )

    assert len(train_loader.dataset) == 3
    assert train_loader.drop_last is True
    assert len(train_loader) == 1
    assert info["train_drop_last"] is True
