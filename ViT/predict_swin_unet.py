"""Save pretrained Swin-UNet qualitative prediction examples."""

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from models.pretrained_swin_unet import SwinUNet
from predict_aspp import run_prediction
from utils.dataset import UAVSegmentationDataset
from utils.experiment import experiment_name_with_epochs
from utils.visualize import get_color_map


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def main():
    num_classes = 12
    image_size = (360, 480)
    batch_size = 1
    decoder_channels = 128
    num_epochs = 50
    max_save = None
    exp_name = experiment_name_with_epochs("swin_unet_pretrained", num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = PROJECT_ROOT / "data" / "raw"
    # checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"{exp_name}_best.pth"
    checkpoint_path = "swin_unet_pretrained_e50_best.pth"
    checkpoint_path = Path(checkpoint_path)
    save_dir = PROJECT_ROOT / "outputs2" / "predictions" / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}\n"
            "Run `python train_swin_unet.py` first."
        )

    test_loader = DataLoader(
        UAVSegmentationDataset(
            image_dir=data_root / "test" / "images",
            label_dir=data_root / "test" / "labels",
            image_size=image_size,
            augment=False,
            image_mean=IMAGE_MEAN,
            image_std=IMAGE_STD,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = SwinUNet(
        num_classes=num_classes,
        pretrained=False,
        decoder_channels=decoder_channels,
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path}")

    run_prediction(
        model=model,
        dataloader=test_loader,
        device=device,
        color_map=get_color_map(num_classes=num_classes),
        save_dir=save_dir,
        max_save=max_save,
    )
    print(f"Prediction figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
