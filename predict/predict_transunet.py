"""Save TransUNet qualitative prediction examples for data/raw/test."""

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.transunet import TransUNet
from predict.predict_aspp import run_prediction
from utils.dataset import UAVSegmentationDataset
from utils.experiment import experiment_name_with_epochs
from utils.visualize import get_color_map


def main():
    num_classes = 12
    image_size = (360, 480)
    batch_size = 1
    base_channels = 32
    transformer_dim = 256
    num_heads = 4
    num_layers = 2
    mlp_dim = 1024
    dropout = 0.1
    num_epochs = 50
    max_save = None
    exp_name = experiment_name_with_epochs("transunet", num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = PROJECT_ROOT / "data" / "raw"
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"{exp_name}_best.pth"
    save_dir = PROJECT_ROOT / "outputs" / "predictions" / exp_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}\n"
            "Run `python train_transunet.py` first."
        )

    test_loader = DataLoader(
        UAVSegmentationDataset(
            image_dir=data_root / "test" / "images",
            label_dir=data_root / "test" / "labels",
            image_size=image_size,
            augment=False,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = TransUNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=base_channels,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        dropout=dropout,
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
