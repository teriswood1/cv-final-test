from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class UAVSegmentationDataset(Dataset):
    """Minimal dataset: yields (tensor, orig_bgr, filename).

    - `image_dir` : Path-like dir containing input images
    - `label_dir` : unused here but kept for API compatibility
    - `image_size` : (H, W) target size
    - `augment` : ignored
    - `image_mean`, `image_std` : used for normalization
    """

    def __init__(
        self,
        image_dir,
        label_dir=None,
        image_size=(360, 480),
        augment=False,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    ):
        self.image_dir = Path(image_dir)
        self.files: List[Path] = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in {'.jpg','.png','.jpeg','.tif','.tiff','.bmp'}])
        self.image_size = image_size
        self.mean = np.array(image_mean, dtype=np.float32)
        self.std = np.array(image_std, dtype=np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        # resize to image_size (H, W)
        h, w = self.image_size
        bgr_resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        return tensor, bgr_resized, str(p)
