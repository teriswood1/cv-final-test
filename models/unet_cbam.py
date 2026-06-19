import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from models.cbam import CBAM


class DoubleConv(nn.Module):
    """
    U-Net 中的基础卷积块。
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class CBAMConvBlock(nn.Module):
    """
    DoubleConv + CBAM。

    先卷积提取特征，再用 CBAM 进行注意力增强。
    """

    def __init__(self, in_channels, out_channels):
        super(CBAMConvBlock, self).__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.cbam = CBAM(channels=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        return x


class Down(nn.Module):
    """
    下采样模块。
    """

    def __init__(self, in_channels, out_channels, use_cbam=True):
        super(Down, self).__init__()

        block = CBAMConvBlock(in_channels, out_channels) if use_cbam else DoubleConv(in_channels, out_channels)

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            block,
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """
    上采样模块。
    """

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_decoder, x_encoder):
        x_decoder = F.interpolate(
            x_decoder,
            size=x_encoder.shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        x = torch.cat([x_encoder, x_decoder], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    """
    输出层。
    """

    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        return self.conv(x)


class UNetCBAM(nn.Module):
    """
    U-Net + CBAM baseline 改进版。

    CBAM 加在 encoder 的卷积块之后。
    """

    def __init__(self, in_channels=3, num_classes=12, base_channels=32):
        super(UNetCBAM, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        # 第一层保持普通卷积，避免浅层过早注意力干扰
        self.inc = DoubleConv(in_channels, base_channels)

        # Encoder 中加入 CBAM
        self.down1 = Down(base_channels, base_channels * 2, use_cbam=True)
        self.down2 = Down(base_channels * 2, base_channels * 4, use_cbam=True)
        self.down3 = Down(base_channels * 4, base_channels * 8, use_cbam=True)
        self.down4 = Down(base_channels * 8, base_channels * 16, use_cbam=True)

        # Decoder
        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2 + base_channels, base_channels)

        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    model = UNetCBAM(in_channels=3, num_classes=12, base_channels=32)
    x = torch.randn(2, 3, 360, 480)
    y = model(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", y.shape)