import pytest
import torch

from models.transunet import TransUNet


def test_transunet_returns_logits_at_input_resolution():
    model = TransUNet(
        in_channels=3,
        num_classes=12,
        base_channels=8,
        transformer_dim=32,
        num_heads=4,
        num_layers=1,
        mlp_dim=64,
    )
    model.eval()

    x = torch.randn(2, 3, 64, 96)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, 12, 64, 96)


def test_transunet_supports_different_input_grid_sizes():
    model = TransUNet(
        in_channels=3,
        num_classes=4,
        base_channels=4,
        transformer_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
    )
    model.eval()

    with torch.no_grad():
        small_logits = model(torch.randn(1, 3, 64, 96))
        large_logits = model(torch.randn(1, 3, 80, 112))

    assert small_logits.shape == (1, 4, 64, 96)
    assert large_logits.shape == (1, 4, 80, 112)


def test_transunet_rejects_feature_grids_larger_than_position_capacity():
    model = TransUNet(
        in_channels=3,
        num_classes=2,
        base_channels=4,
        transformer_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        max_position_embeddings=4,
    )

    with pytest.raises(ValueError, match="max_position_embeddings"):
        model(torch.randn(1, 3, 64, 96))
