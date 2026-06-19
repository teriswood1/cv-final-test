import pytest
import torch

from utils.metrics import SegmentationMetric


def test_metric_ignores_configured_label_id():
    metric = SegmentationMetric(num_classes=3, ignore_index=2)

    labels = torch.tensor([[[0, 1], [2, 2]]])
    preds = torch.tensor([[[0, 0], [1, 2]]])

    metric.update(preds, labels)
    results = metric.get_results()

    assert results["Confusion Matrix"].sum() == 2
    assert results["Pixel Accuracy"] == pytest.approx(0.5)
    assert results["Class IoU"].tolist() == pytest.approx([0.5, 0.0])
    assert results["Mean IoU"] == pytest.approx(0.25)
