import torch
import torch.nn as nn

from models.unet import DoubleConv, Down, OutConv, Up


class FeatureMapTransformer(nn.Module):
    """Apply Transformer self-attention over a CNN feature map."""

    def __init__(
        self,
        in_channels,
        transformer_dim=256,
        num_heads=4,
        num_layers=2,
        mlp_dim=1024,
        dropout=0.1,
        max_position_embeddings=2048,
    ):
        super().__init__()
        if transformer_dim % num_heads != 0:
            raise ValueError("transformer_dim must be divisible by num_heads")
        if max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")

        self.max_position_embeddings = max_position_embeddings
        self.input_projection = nn.Conv2d(in_channels, transformer_dim, kernel_size=1)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, max_position_embeddings, transformer_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(transformer_dim)
        self.output_projection = nn.Conv2d(transformer_dim, in_channels, kernel_size=1)

        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x):
        projected = self.input_projection(x)
        batch_size, channels, height, width = projected.shape
        token_count = height * width
        if token_count > self.max_position_embeddings:
            raise ValueError(
                "Feature grid requires "
                f"{token_count} position embeddings, but max_position_embeddings="
                f"{self.max_position_embeddings}"
            )

        tokens = projected.flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embeddings[:, :token_count, :]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)

        features = tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )
        return x + self.output_projection(features)


class TransUNet(nn.Module):
    """
    U-Net encoder-decoder with a Transformer bottleneck.

    The CNN path keeps U-Net's local detail modeling, while the Transformer bridge
    lets the bottleneck exchange global context before decoding.
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=12,
        base_channels=32,
        transformer_dim=256,
        num_heads=4,
        num_layers=2,
        mlp_dim=1024,
        dropout=0.1,
        max_position_embeddings=2048,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.transformer_dim = transformer_dim
        bottleneck_channels = base_channels * 16

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, bottleneck_channels)
        self.transformer = FeatureMapTransformer(
            in_channels=bottleneck_channels,
            transformer_dim=transformer_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
        )

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

        x = self.transformer(x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
