import torch

from models.deeplab import build_deeplabv3_resnet50


def test_deeplab_builder_replaces_classifier_head_channels():
    model = build_deeplabv3_resnet50(num_classes=12, pretrained=False)
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(1, 3, 64, 64))

    assert logits.shape == (1, 12, 64, 64)
