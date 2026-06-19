import torch

from models.pretrained_swin_unet import SwinUNet


def test_swin_unet_returns_logits_at_input_resolution():
    model = SwinUNet(
        num_classes=12,
        pretrained=False,
        decoder_channels=32,
    )
    model.eval()

    x = torch.randn(2, 3, 64, 96)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 12, 64, 96)


def test_swin_unet_supports_non_square_input():
    model = SwinUNet(
        num_classes=4,
        pretrained=False,
        decoder_channels=16,
    )
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, 80, 112))

    assert logits.shape == (1, 4, 80, 112)
