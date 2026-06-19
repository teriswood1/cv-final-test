from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class UAVSegmentationDataset(Dataset):
    """
    无人机语义分割数据集读取类。

    功能：
    1. 读取 RGB 原图；
    2. 读取单通道 label 图；
    3. 将 image 转成 FloatTensor，形状为 [3, H, W]；
    4. 将 label 转成 LongTensor，形状为 [H, W]；
    5. 支持基础数据增强。
    """

    def __init__(
        self,
        image_dir,
        label_dir,
        image_size=(360, 480),
        augment=False,
        augment_mode="strong",
        num_label_ids=None,
        image_mean=None,
        image_std=None,
    ):
        """
        参数说明：
        image_dir: 原图文件夹路径
        label_dir: 标签文件夹路径
        image_size: 输入尺寸，格式为 (height, width)
                    你的原图是 480×360，也就是 width=480, height=360
                    所以这里默认写 (360, 480)
        augment: 是否使用数据增强
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = image_size
        self.augment = augment
        self.augment_mode = augment_mode
        self.num_label_ids = num_label_ids
        self.image_mean = image_mean
        self.image_std = image_std
        if (self.image_mean is None) != (self.image_std is None):
            raise ValueError("image_mean and image_std must be configured together")

        self.image_paths = self._collect_image_paths()

        if len(self.image_paths) == 0:
            raise RuntimeError(f"没有在 {self.image_dir} 中找到有效图片")

    def _is_valid_file(self, path: Path):
        """
        过滤无效文件，比如 .DS_Store、._xxx.png 等。
        """
        valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        if path.name.startswith("."):
            return False
        if path.name.startswith("._"):
            return False
        if path.suffix.lower() not in valid_suffixes:
            return False

        return True

    def _collect_image_paths(self):
        """
        收集所有图片路径，并确保对应 label 存在。
        """
        image_paths = [
            p for p in self.image_dir.iterdir()
            if p.is_file() and self._is_valid_file(p)
        ]

        image_paths = sorted(image_paths, key=lambda x: x.name)

        valid_image_paths = []
        missing_labels = []

        for image_path in image_paths:
            label_path = self.label_dir / image_path.name

            if label_path.exists():
                valid_image_paths.append(image_path)
            else:
                missing_labels.append(image_path.name)

        if missing_labels:
            print(f"[警告] 有 {len(missing_labels)} 张图片找不到对应 label，例如：")
            print(missing_labels[:10])

        return valid_image_paths

    def __len__(self):
        return len(self.image_paths)

    def _load_image_and_label(self, image_path: Path):
        """
        读取一张图片和对应标签。
        """
        label_path = self.label_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        return image, label

    def _resize(self, image, label):
        """
        调整图片和标签大小。

        注意：
        image 使用双线性插值；
        label 必须使用最近邻插值，否则类别编号会被插坏。
        """
        image = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BILINEAR)
        label = TF.resize(label, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        return image, label

    def _augment(self, image, label):
        """
        训练集数据增强（image 与 label 几何变换保持同步）。
        """
        if self.augment_mode == "road":
            return self._road_augment(image, label)

        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        if torch.rand(1).item() < 0.2:
            image = TF.vflip(image)
            label = TF.vflip(label)

        if self.augment_mode == "light":
            return image, label

        if torch.rand(1).item() < 0.3:
            angle = float(torch.empty(1).uniform_(-10, 10).item())
            image = TF.rotate(
                image,
                angle,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            )
            label = TF.rotate(
                label,
                angle,
                interpolation=TF.InterpolationMode.NEAREST,
                fill=0,
            )

        if torch.rand(1).item() < 0.5:
            image = TF.adjust_brightness(image, brightness_factor=0.9 + torch.rand(1).item() * 0.2)
            image = TF.adjust_contrast(image, contrast_factor=0.9 + torch.rand(1).item() * 0.2)

        return image, label

    def _road_augment(self, image, label):
        """
        面向道路场景的保守增强：横向翻转、放大裁剪、轻颜色扰动。
        """
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        scale = 1.0 + torch.rand(1).item() * 0.25
        target_h, target_w = self.image_size
        scaled_h = max(target_h, int(round(target_h * scale)))
        scaled_w = max(target_w, int(round(target_w * scale)))

        if scaled_h != target_h or scaled_w != target_w:
            image = TF.resize(
                image,
                (scaled_h, scaled_w),
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            label = TF.resize(
                label,
                (scaled_h, scaled_w),
                interpolation=TF.InterpolationMode.NEAREST,
            )

            max_top = scaled_h - target_h
            max_left = scaled_w - target_w
            top = int(torch.randint(0, max_top + 1, (1,)).item()) if max_top else 0
            left = int(torch.randint(0, max_left + 1, (1,)).item()) if max_left else 0
            image = TF.crop(image, top, left, target_h, target_w)
            label = TF.crop(label, top, left, target_h, target_w)

        if torch.rand(1).item() < 0.5:
            brightness = 0.9 + torch.rand(1).item() * 0.2
            contrast = 0.9 + torch.rand(1).item() * 0.2
            image = TF.adjust_brightness(image, brightness_factor=brightness)
            image = TF.adjust_contrast(image, contrast_factor=contrast)

        return image, label

    def _to_tensor(self, image, label):
        """
        转换成 PyTorch Tensor。
        """
        # image: PIL RGB -> FloatTensor [3, H, W]，范围 0~1
        image = TF.to_tensor(image)
        if self.image_mean is not None:
            image = TF.normalize(image, mean=self.image_mean, std=self.image_std)

        # label: PIL L -> numpy [H, W] -> LongTensor [H, W]
        label = np.array(label, dtype=np.int64)
        label = torch.from_numpy(label).long()

        return image, label

    def _validate_label_ids(self, label, image_path):
        if self.num_label_ids is None:
            return

        invalid = label[(label < 0) | (label >= self.num_label_ids)]
        if invalid.numel() == 0:
            return

        invalid_ids = sorted(invalid.unique().tolist())
        raise ValueError(
            f"Label ids {invalid_ids} in {image_path.name} are outside configured "
            f"label ids [0, {self.num_label_ids - 1}]"
        )

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image, label = self._load_image_and_label(image_path)

        image, label = self._resize(image, label)

        if self.augment:
            image, label = self._augment(image, label)

        image, label = self._to_tensor(image, label)
        self._validate_label_ids(label, image_path)

        return {
            "image": image,
            "label": label,
            "filename": image_path.name,
        }
