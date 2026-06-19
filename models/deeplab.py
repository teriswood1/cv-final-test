import torch.nn as nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


class DeepLabLogits(nn.Module):
    """Expose Torchvision DeepLab output as a logits tensor."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def _replace_classifier_head(model, num_classes):
    final_conv = model.classifier[-1]
    model.classifier[-1] = nn.Conv2d(
        final_conv.in_channels,
        num_classes,
        kernel_size=final_conv.kernel_size,
        stride=final_conv.stride,
    )
    model.aux_classifier = None
    return model


def build_deeplabv3_resnet50(num_classes, pretrained=True):
    """Build DeepLabV3-ResNet50 and adapt its classifier to local mask ids."""
    if pretrained:
        model = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT,
        )
    else:
        model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
            aux_loss=False,
        )

    return DeepLabLogits(_replace_classifier_head(model, num_classes))
