"""
通用训练 / 验证循环。
"""

import time

import torch
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes, metric_class):
    model.eval()
    total_loss = 0.0
    metric = metric_class(num_classes=num_classes)
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in progress_bar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        metric.update(outputs, labels)

    results = metric.get_results()
    return (
        total_loss / len(dataloader),
        results["Pixel Accuracy"],
        results["Mean IoU"],
        results["Class IoU"],
    )


def run_training(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_classes,
    num_epochs,
    best_model_path,
    last_model_path,
    log_path,
    metric_class,
    scheduler=None,
    checkpoint_extra=None,
    start_epoch=1,
    best_miou=0.0,
    append_log=False,
):
    """完整训练流程，返回 best_miou。"""
    if start_epoch < 1:
        raise ValueError("start_epoch must be positive")
    if start_epoch > num_epochs + 1:
        raise ValueError("start_epoch cannot exceed num_epochs + 1")

    checkpoint_extra = checkpoint_extra or {}

    log_mode = "a" if append_log else "w"
    with open(log_path, log_mode, encoding="utf-8") as f:
        if not append_log:
            f.write("epoch,train_loss,val_loss,pixel_acc,miou,lr\n")

    start_time = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, pixel_acc, miou, _ = evaluate(
            model, val_loader, criterion, device, num_classes, metric_class
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Pixel Acc:  {pixel_acc:.4f}")
        print(f"mIoU:       {miou:.4f}")

        current_lr = optimizer.param_groups[0]["lr"]
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                f"{pixel_acc:.6f},{miou:.6f},{current_lr:.8f}\n"
            )

        if scheduler is not None:
            scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "miou": miou,
            "pixel_acc": pixel_acc,
            "lr": current_lr,
            "num_classes": num_classes,
            **checkpoint_extra,
        }
        torch.save(checkpoint, last_model_path)

        if miou > best_miou:
            best_miou = miou
            torch.save(checkpoint, best_model_path)
            print(f"[保存最优模型] mIoU 提升到 {best_miou:.4f}")
            print(f"模型保存路径：{best_model_path}")

    print("=" * 80)
    print("训练完成")
    print(f"最佳验证集 mIoU：{best_miou:.4f}")
    print(f"最优模型路径：{best_model_path}")
    print(f"训练总耗时：{(time.time() - start_time) / 60:.2f} 分钟")
    print("=" * 80)

    return best_miou
