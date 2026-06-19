import sys
from pathlib import Path

import torch
import torch.nn as nn

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from models.aspp import ASPP
from models.unet_cbam import DoubleConv, Down, OutConv, Up


class UNetCBAMASPP(nn.Module):
    """
    U-Net + CBAM + ASPP 完整改进版。

    - CBAM：加在 encoder 的 down1~down4 之后（inc 不加）
    - ASPP：加在 encoder 最底层输出（bottleneck）之后、decoder 之前
    """

    def __init__(self, in_channels=3, num_classes=12, base_channels=32):
        super(UNetCBAMASPP, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        bottleneck_channels = base_channels * 16

        self.inc = DoubleConv(in_channels, base_channels)

        self.down1 = Down(base_channels, base_channels * 2, use_cbam=True)
        self.down2 = Down(base_channels * 2, base_channels * 4, use_cbam=True)
        self.down3 = Down(base_channels * 4, base_channels * 8, use_cbam=True)
        self.down4 = Down(base_channels * 8, bottleneck_channels, use_cbam=True)

        self.aspp = ASPP(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            atrous_rates=(6, 12, 18),
        )

        self.up1 = Up(bottleneck_channels + base_channels * 8, base_channels * 8)
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

        x5 = self.aspp(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


if __name__ == "__main__":
    model = UNetCBAMASPP(in_channels=3, num_classes=12, base_channels=32)
    x = torch.randn(2, 3, 360, 480)
    y = model(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", y.shape)
    print(
        "参数量:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
