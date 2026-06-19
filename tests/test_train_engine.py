import pytest
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from utils.train_engine import run_training


class TinyMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def update(self, outputs, labels):
        return None

    def get_results(self):
        return {"Pixel Accuracy": 1.0, "Mean IoU": 1.0, "Class IoU": [1.0]}


def batch():
    return [
        {
            "image": torch.ones(1, 1, 2, 2),
            "label": torch.zeros(1, 2, 2, dtype=torch.long),
        }
    ]


def test_run_training_steps_scheduler_and_logs_lr(tmp_path):
    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    optimizer = SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    run_training(
        model=model,
        train_loader=batch(),
        val_loader=batch(),
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device="cpu",
        num_classes=1,
        num_epochs=1,
        best_model_path=tmp_path / "best.pth",
        last_model_path=tmp_path / "last.pth",
        log_path=tmp_path / "log.csv",
        metric_class=TinyMetric,
        checkpoint_extra={"model_name": "tiny"},
    )

    log_text = (tmp_path / "log.csv").read_text()
    checkpoint = torch.load(
        tmp_path / "last.pth",
        map_location="cpu",
        weights_only=False,
    )

    assert "lr" in log_text.splitlines()[0]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert checkpoint["model_name"] == "tiny"


def test_run_training_resumes_without_rewriting_existing_log(tmp_path):
    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    optimizer = SGD(model.parameters(), lr=0.1)
    log_path = tmp_path / "log.csv"
    log_path.write_text(
        "epoch,train_loss,val_loss,pixel_acc,miou,lr\n"
        "1,2.000000,1.500000,0.500000,0.500000,0.10000000\n",
        encoding="utf-8",
    )

    best_miou = run_training(
        model=model,
        train_loader=batch(),
        val_loader=batch(),
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device="cpu",
        num_classes=1,
        num_epochs=2,
        best_model_path=tmp_path / "best.pth",
        last_model_path=tmp_path / "last.pth",
        log_path=log_path,
        metric_class=TinyMetric,
        checkpoint_extra={"model_name": "tiny"},
        start_epoch=2,
        best_miou=0.5,
        append_log=True,
    )

    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    checkpoint = torch.load(
        tmp_path / "last.pth",
        map_location="cpu",
        weights_only=False,
    )

    assert len(log_lines) == 3
    assert log_lines[1].startswith("1,")
    assert log_lines[2].startswith("2,")
    assert checkpoint["epoch"] == 2
    assert best_miou == pytest.approx(1.0)
