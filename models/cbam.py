import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    通道注意力模块。

    作用：
    判断特征图中哪些通道更重要。
    """

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()

        hidden_channels = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


class SpatialAttention(nn.Module):
    """
    空间注意力模块。

    作用：
    判断特征图中哪些空间位置更重要。
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel_size 只能是 3 或 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)

        return x * attention


class CBAM(nn.Module):
    """
    CBAM 注意力模块。

    顺序：
    输入特征
      ↓
    通道注意力
      ↓
    空间注意力
      ↓
    输出增强特征
    """

    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=reduction,
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel_size,
        )

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 64, 360, 480)

    cbam = CBAM(channels=64)
    y = cbam(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", y.shape)