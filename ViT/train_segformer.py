import csv
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from models.vehicle_scene_relation_attention import add_vehicle_scene_relation_attention


CLASS_NAMES = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
    "Unlabelled",
]
NUM_CLASSES = 12
IGNORE_INDEX = 255
IMAGE_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

# ==== 可选参数（按需手动修改） ====
TRAIN_IMAGES = "train/images"
TRAIN_LABELS = "train/labels"
VAL_IMAGES = "test/images"
VAL_LABELS = "test/labels"
BATCH_SIZE = 4
VAL_BATCH_SIZE = 1
NUM_WORKERS = 4
EPOCHS = 50
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.01
PATCH_SIZE = 256
PATCHES_PER_IMAGE = 4
SEED = 42
OUTPUT_DIR = "outputs/segformer_camvid12_vsra"
SAVE_BEST_ONLY = False
ENABLE_RELATION_ATTENTION = True
RELATION_CHANNELS = 64
RELATION_POOL_SIZE = 16


def set_seed(seed: int) -> None:
    # 固定随机种子以保证复现实验结果
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    train_images: str
    train_labels: str
    val_images: str
    val_labels: str
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float
    patch_size: int
    patches_per_image: int
    seed: int
    device: str
    output_dir: str
    save_best_only: bool


class UAVDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        patch_size: int = 512,
        patches_per_image: int = 4,
        is_train: bool = True,
        transforms: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.is_train = is_train
        self.transforms = transforms

        self.image_paths = self._scan_images(self.image_dir)
        if not self.image_paths:
            raise ValueError(f"未在目录中找到图像: {self.image_dir}")
        self._warn_if_patch_may_degenerate()

    def _scan_images(self, root: str) -> List[str]:
        # 仅扫描常见图像后缀，保证与标签文件一一对应
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        paths = []
        for name in os.listdir(root):
            if os.path.splitext(name)[1].lower() in exts:
                paths.append(os.path.join(root, name))
        paths.sort()
        return paths

    def _get_mask_path(self, image_path: str) -> str:
        # 通过同名文件匹配标签图（支持多种后缀）
        base = os.path.splitext(os.path.basename(image_path))[0]
        for ext in [".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"]:
            candidate = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"未找到图像对应的标签: {image_path}")

    def __len__(self) -> int:
        # 每张图像采样多个 patch，扩大训练样本数
        if self.is_train:
            return len(self.image_paths) * self.patches_per_image
        return len(self.image_paths)

    def _warn_if_patch_may_degenerate(self) -> None:
        if not self.is_train:
            return
        sample = self._read_image(self.image_paths[0])
        h, w = sample.shape[:2]
        if self.patch_size >= h and self.patch_size >= w:
            print(
                f"警告: PATCH_SIZE={self.patch_size} 大于等于样本尺寸 {w}x{h}，"
                "随机裁剪会退化为固定位置。"
            )
        elif self.patch_size >= h or self.patch_size >= w:
            print(
                f"提示: PATCH_SIZE={self.patch_size} 接近样本尺寸 {w}x{h}，"
                "随机裁剪自由度有限。"
            )

    def _random_patch(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 训练时从大图随机裁剪 512x512 patch，降低显存占用并提升多样性
        h, w = image.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=IGNORE_INDEX)
            h, w = image.shape[:2]
        y = random.randint(0, h - ps)
        x = random.randint(0, w - ps)
        return image[y : y + ps, x : x + ps], mask[y : y + ps, x : x + ps]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # idx 映射到具体图像，再做随机裁剪
        if self.is_train:
            image_idx = idx // self.patches_per_image
        else:
            image_idx = idx

        image_path = self.image_paths[image_idx]
        mask_path = self._get_mask_path(image_path)
        image = self._read_image(image_path)
        mask = self._read_mask(mask_path)

        if self.is_train:
            image, mask = self._random_patch(image, mask)

        if self.is_train and self.transforms:
            image, mask = self._augment(image, mask)
            # 强制转为 long，满足 CrossEntropyLoss 的标签类型要求
        if True:
            # 默认将图像转为 CHW，并归一化到 [0, 1]
            image = self._to_tensor(image)
            mask = torch.from_numpy(mask).long()

        return image, mask

    def _read_image(self, path: str) -> np.ndarray:
        image = Image.open(path).convert("RGB")

        # OpenCV 读入为 BGR，这里转为 RGB
        return np.array(image)
        if image is None:
            raise FileNotFoundError(f"读取图像失败: {path}")

    def _read_mask(self, path: str) -> np.ndarray:
        mask_image = Image.open(path).convert("L")

        mask = np.array(mask_image, dtype=np.int64)
        if mask is None:
            raise FileNotFoundError(f"读取标签失败: {path}")
        # 非法标签统一映射为 ignore_index，避免干扰训练与评估
        mask[(mask < 0) | (mask >= NUM_CLASSES)] = IGNORE_INDEX
        return mask

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        if random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        if random.random() < 0.5:
            pil_image = Image.fromarray(image)
            pil_image = ImageEnhance.Brightness(pil_image).enhance(random.uniform(0.8, 1.2))
            pil_image = ImageEnhance.Contrast(pil_image).enhance(random.uniform(0.8, 1.2))
            pil_image = ImageEnhance.Color(pil_image).enhance(random.uniform(0.8, 1.2))
            image = np.array(pil_image)

        return image, mask

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGE_MEAN) / IMAGE_STD
        return torch.from_numpy(image.transpose(2, 0, 1)).float()


def get_train_transforms() -> bool:
    # 训练增强：旋转/翻转/颜色扰动 + ImageNet 归一化
    return True


def get_val_transforms() -> bool:
    # 验证只做归一化，保证评估稳定
    return False


def get_segformer_model(pretrained: bool = True) -> nn.Module:
    if pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b1",
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
            local_files_only=True,
        )
    else:
        config = SegformerConfig.from_pretrained(
            "nvidia/mit-b1",
            num_labels=NUM_CLASSES,
            local_files_only=True,
        )
        model = SegformerForSemanticSegmentation(config)
    model.config.num_labels = NUM_CLASSES
    # 告诉 SegFormer 训练/评估时忽略背景标签
    model.config.semantic_loss_ignore_index = IGNORE_INDEX
    if ENABLE_RELATION_ATTENTION:
        model = add_vehicle_scene_relation_attention(
            model,
            relation_channels=RELATION_CHANNELS,
            pool_size=RELATION_POOL_SIZE,
        )
    return model


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W]，targets: [B, H, W]
        num_classes = self.num_classes
        valid = targets != self.ignore_index
        if valid.sum() == 0:
            return torch.zeros([], device=logits.device, dtype=logits.dtype)

        targets = targets.clone()
        targets[~valid] = 0

        probs = F.softmax(logits, dim=1)
        # 仅对有效像素计算 Dice
        probs = probs.permute(0, 2, 3, 1).contiguous()
        probs = probs.view(-1, num_classes)
        targets_flat = targets.view(-1)
        valid_flat = valid.view(-1)

        probs = probs[valid_flat]
        targets_flat = targets_flat[valid_flat]

        targets_onehot = F.one_hot(targets_flat, num_classes=num_classes).float()
        intersection = torch.sum(probs * targets_onehot, dim=0)
        cardinality = torch.sum(probs + targets_onehot, dim=0)
        dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        return 1.0 - dice.mean()


class HybridLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 交叉熵处理分类稳态，Dice 强化小目标类别
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return ce_loss + dice_loss


def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    # 过滤 ignore_index 后再统计混淆矩阵
    preds = preds.view(-1)
    targets = targets.view(-1)
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    if preds.numel() == 0:
        return torch.zeros((num_classes, num_classes), device=preds.device, dtype=torch.int64)

    cm = torch.bincount(
        num_classes * targets + preds, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    return cm


def compute_metrics_from_cm(cm: torch.Tensor) -> Tuple[float, List[float], float]:
    # IoU = TP / (TP + FP + FN)，PA = sum(TP) / 全像素
    diag = torch.diag(cm).float()
    denom = cm.sum(dim=1).float() + cm.sum(dim=0).float() - diag
    iou = torch.where(denom > 0, diag / denom, torch.zeros_like(denom))
    miou = iou.mean().item()
    pixel_acc = diag.sum().item() / (cm.sum().item() + 1e-6)
    return pixel_acc, iou.tolist(), miou


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    device: str,
) -> float:
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        # SegFormer 输入为 [B, 3, H, W]，标签为 [B, H, W]
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(pixel_values=images)
        logits = outputs.logits
        # 将 logits 上采样到标签尺寸
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(logits, masks)
        # 标准反向传播流程
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    scheduler.step()
    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, List[float], float]:
    model.eval()
    running_loss = 0.0
    cm_total = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=device, dtype=torch.int64)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(pixel_values=images)
            logits = outputs.logits
            # 将 logits 上采样到标签尺寸
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            # 逐批次累计混淆矩阵
            cm = compute_confusion_matrix(preds, masks, NUM_CLASSES, IGNORE_INDEX)
            cm_total += cm

    pixel_acc, iou_list, miou = compute_metrics_from_cm(cm_total)
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, pixel_acc, iou_list, miou


def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    # 训练集每图多 patch，验证集每图 1 个 patch
    train_dataset = UAVDataset(
        cfg.train_images,
        cfg.train_labels,
        patch_size=cfg.patch_size,
        patches_per_image=cfg.patches_per_image,
        is_train=True,
        transforms=get_train_transforms(),
    )
    val_dataset = UAVDataset(
        cfg.val_images,
        cfg.val_labels,
        patch_size=cfg.patch_size,
        patches_per_image=1,
        is_train=False,
        transforms=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def format_iou(iou_list: List[float]) -> str:
    parts = [f"{name}:{iou_list[i]:.4f}" for i, name in enumerate(CLASS_NAMES)]
    return " | ".join(parts)


def get_config() -> TrainConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 将命令行参数封装为配置对象
    return TrainConfig(
        train_images=TRAIN_IMAGES,
        train_labels=TRAIN_LABELS,
        val_images=VAL_IMAGES,
        val_labels=VAL_LABELS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patch_size=PATCH_SIZE,
        patches_per_image=PATCHES_PER_IMAGE,
        seed=SEED,
        device=device,
        output_dir=OUTPUT_DIR,
        save_best_only=SAVE_BEST_ONLY,
    )


def main() -> None:
    cfg = get_config()
    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "train_log.csv")
    best_model_path = os.path.join(cfg.output_dir, "best_model.pth")
    last_model_path = os.path.join(cfg.output_dir, "last_model.pth")
    loss_curve_path = os.path.join(cfg.output_dir, "loss_curve.png")
    miou_curve_path = os.path.join(cfg.output_dir, "miou_curve.png")

    # 构建数据与模型
    train_loader, val_loader = build_loaders(cfg)

    resume_available = os.path.exists(last_model_path)
    model = get_segformer_model(pretrained=not resume_available).to(cfg.device)
    criterion = HybridLoss(NUM_CLASSES, IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history_epochs: List[int] = []
    history_train_loss: List[float] = []
    history_val_loss: List[float] = []
    history_pa: List[float] = []
    history_miou: List[float] = []
    history_iou: List[List[float]] = []
    best_miou = -1.0
    start_epoch = 1
    append_log = False

    if os.path.exists(log_path):
        with open(log_path, mode="r", newline="", encoding="utf-8") as log_file:
            reader = csv.reader(log_file)
            header = next(reader, None)
            for row in reader:
                if len(row) < 5:
                    continue
                history_epochs.append(int(row[0]))
                history_train_loss.append(float(row[1]))
                history_val_loss.append(float(row[2]))
                history_pa.append(float(row[3]))
                history_miou.append(float(row[4]))
                history_iou.append([float(value) for value in row[5:]])
        if history_miou:
            best_miou = max(history_miou)

    if os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path, map_location=cfg.device)
        if checkpoint.get("num_classes") != NUM_CLASSES:
            raise ValueError(
                f"Checkpoint num_classes={checkpoint.get('num_classes')} "
                f"does not match current NUM_CLASSES={NUM_CLASSES}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            for _ in range(int(checkpoint.get("epoch", 0))):
                scheduler.step()
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        append_log = os.path.exists(log_path)
        print(f"Resuming from epoch {start_epoch}/{cfg.epochs}; best mIoU={best_miou:.4f}")

    if start_epoch > cfg.epochs:
        print(f"Training already reached epoch {start_epoch - 1}/{cfg.epochs}.")
        return

    if not append_log:
        with open(log_path, mode="w", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            header = [
                "epoch",
                "train_loss",
                "val_loss",
                "pa",
                "miou",
            ] + [f"iou_{name.lower()}" for name in CLASS_NAMES]
            writer.writerow(header)

    for epoch in range(start_epoch, cfg.epochs + 1):
        # 每个 epoch 输出训练/验证损失与分割指标
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, cfg.device)
        val_loss, pixel_acc, iou_list, miou = evaluate(model, val_loader, criterion, cfg.device)

        print(
            f"轮次 {epoch:03d}/{cfg.epochs} | "
            f"训练 Loss: {train_loss:.4f} | "
            f"验证 Loss: {val_loss:.4f} | "
            f"PA: {pixel_acc:.4f} | "
            f"mIoU: {miou:.4f} | "
            f"IoU: {format_iou(iou_list)}"
        )

        history_epochs.append(epoch)
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_pa.append(pixel_acc)
        history_miou.append(miou)
        history_iou.append(iou_list)

        with open(log_path, mode="a", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([epoch, train_loss, val_loss, pixel_acc, miou] + iou_list)

        if miou > best_miou:
            best_miou = miou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "miou": miou,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "relation_attention": ENABLE_RELATION_ATTENTION,
                    "relation_channels": RELATION_CHANNELS,
                    "relation_pool_size": RELATION_POOL_SIZE,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": cfg.__dict__,
                },
                best_model_path,
            )

        if not cfg.save_best_only:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "miou": miou,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "relation_attention": ENABLE_RELATION_ATTENTION,
                    "relation_channels": RELATION_CHANNELS,
                    "relation_pool_size": RELATION_POOL_SIZE,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": cfg.__dict__,
                },
                last_model_path,
            )

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(history_epochs, history_train_loss, label="Train Loss")
        plt.plot(history_epochs, history_val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_curve_path, dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(history_epochs, history_miou, label="mIoU")
        plt.xlabel("Epoch")
        plt.ylabel("mIoU")
        plt.title("Validation mIoU")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(miou_curve_path, dpi=150)
        plt.close()
    except ImportError:
        print("未安装 matplotlib，已跳过训练曲线保存。")


if __name__ == "__main__":
    main()
