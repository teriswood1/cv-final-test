import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_T_Weights, swin_t


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SwinEncoder(nn.Module):
    """Expose Swin-T feature maps as NCHW tensors at four scales."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        self.backbone = swin_t(weights=weights)
        self.features = self.backbone.features
        self.out_channels = (96, 192, 384, 768)

    @staticmethod
    def _to_nchw(x):
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        outputs = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in {1, 3, 5, 7}:
                outputs.append(self._to_nchw(x))
        return outputs


class FPNDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, num_classes):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(channels, decoder_channels, kernel_size=1)
                for channels in encoder_channels
            ]
        )
        self.smooth_convs = nn.ModuleList(
            [ConvBNReLU(decoder_channels, decoder_channels) for _ in encoder_channels]
        )
        self.head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, features, output_size):
        laterals = [
            conv(feature) for conv, feature in zip(self.lateral_convs, features)
        ]
        x = laterals[-1]
        decoded = [None] * len(laterals)
        decoded[-1] = self.smooth_convs[-1](x)

        for index in range(len(laterals) - 2, -1, -1):
            x = F.interpolate(
                x,
                size=laterals[index].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            x = x + laterals[index]
            decoded[index] = self.smooth_convs[index](x)

        logits = self.head(decoded[0])
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class SwinUNet(nn.Module):
    """ImageNet-pretrained Swin-T encoder with a lightweight FPN decoder."""

    def __init__(
        self,
        num_classes=12,
        pretrained=True,
        decoder_channels=128,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.encoder = SwinEncoder(pretrained=pretrained)
        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        output_size = x.shape[2:]
        features = self.encoder(x)
        return self.decoder(features, output_size)
