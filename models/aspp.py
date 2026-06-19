import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Module):
    """空洞卷积分支。"""

    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling（DeepLab 风格）。

    五个分支：1×1 卷积、三个不同膨胀率的 3×3 空洞卷积、全局平均池化。
    拼接后再用 1×1 卷积映射回 out_channels。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        atrous_rates=(6, 12, 18),
        aspp_channels=None,
        dropout=0.1,
    ):
        super(ASPP, self).__init__()

        if aspp_channels is None:
            aspp_channels = max(in_channels // 4, 32)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aspp_channels = aspp_channels

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, aspp_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
        )

        self.atrous_convs = nn.ModuleList(
            [
                ASPPConv(in_channels, aspp_channels, dilation=rate)
                for rate in atrous_rates
            ]
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, aspp_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
        )

        num_branches = 2 + len(atrous_rates)  # 1x1 + dilated + global
        self.project = nn.Sequential(
            nn.Conv2d(
                aspp_channels * num_branches,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        branches = [self.conv1x1(x)]

        for atrous_conv in self.atrous_convs:
            branches.append(atrous_conv(x))

        pool = self.global_pool(x)
        pool = F.interpolate(
            pool,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=True,
        )
        branches.append(pool)

        x = torch.cat(branches, dim=1)
        return self.project(x)


if __name__ == "__main__":
    # bottleneck 特征图尺寸：360x480 经 4 次下采样 -> 22x30
    x = torch.randn(2, 512, 22, 30)
    aspp = ASPP(in_channels=512, out_channels=512)
    y = aspp(x)
    print("输入 shape:", x.shape)
    print("输出 shape:", y.shape)
