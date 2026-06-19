import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    U-Net 中最基础的卷积块：
    Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    下采样模块：
    MaxPool2d 负责把特征图尺寸减半；
    DoubleConv 负责提取更深层特征。
    """

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """
    上采样模块：
    先用双线性插值把特征图尺寸放大；
    再和编码器对应层的特征拼接；
    最后用 DoubleConv 融合特征。
    """

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_decoder, x_encoder):
        # 将 decoder 的特征图上采样到 encoder 特征图的大小
        x_decoder = F.interpolate(
            x_decoder,
            size=x_encoder.shape[2:],
            mode="bilinear",
            align_corners=True
        )

        # 在通道维度拼接
        x = torch.cat([x_encoder, x_decoder], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    """
    最后一层 1×1 卷积：
    将通道数转换为类别数。
    """

    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    标准 U-Net baseline。

    输入：
        x: [B, 3, H, W]

    输出：
        logits: [B, num_classes, H, W]
    """

    def __init__(self, in_channels=3, num_classes=12, base_channels=64):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)              # 3 -> 64
        self.down1 = Down(base_channels, base_channels * 2)            # 64 -> 128
        self.down2 = Down(base_channels * 2, base_channels * 4)        # 128 -> 256
        self.down3 = Down(base_channels * 4, base_channels * 8)        # 256 -> 512
        self.down4 = Down(base_channels * 8, base_channels * 16)       # 512 -> 1024

        # Decoder
        self.up1 = Up(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2 + base_channels, base_channels)

        # Output
        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)       # [B, 64, H, W]
        x2 = self.down1(x1)    # [B, 128, H/2, W/2]
        x3 = self.down2(x2)    # [B, 256, H/4, W/4]
        x4 = self.down3(x3)    # [B, 512, H/8, W/8]
        x5 = self.down4(x4)    # [B, 1024, H/16, W/16]

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits


if __name__ == "__main__":
    # 简单测试模型输入输出尺寸
    model = UNet(in_channels=3, num_classes=12, base_channels=64)

    x = torch.randn(2, 3, 360, 480)
    y = model(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", y.shape)