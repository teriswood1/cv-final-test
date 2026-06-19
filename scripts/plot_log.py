from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def plot_training_log(log_path, save_dir, model_name="unet_baseline"):
    """
    根据 train.py 保存的训练日志画曲线。

    日志格式：
    epoch,train_loss,val_loss,pixel_acc,miou
    """
    log_path = Path(log_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        raise FileNotFoundError(f"找不到日志文件：{log_path}")

    df = pd.read_csv(log_path)

    print("日志内容预览：")
    print(df.head())

    # =========================
    # 1. Train Loss / Val Loss
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name}: Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    loss_save_path = save_dir / f"{model_name}_loss_curve.png"
    plt.savefig(loss_save_path, dpi=200, bbox_inches="tight")
    plt.close()

    # =========================
    # 2. Pixel Accuracy
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["pixel_acc"], marker="o", label="Pixel Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Pixel Accuracy")
    plt.title(f"{model_name}: Validation Pixel Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    pa_save_path = save_dir / f"{model_name}_pixel_accuracy_curve.png"
    plt.savefig(pa_save_path, dpi=200, bbox_inches="tight")
    plt.close()

    # =========================
    # 3. mIoU
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["miou"], marker="o", label="mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title(f"{model_name}: Validation mIoU")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    miou_save_path = save_dir / f"{model_name}_miou_curve.png"
    plt.savefig(miou_save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("=" * 80)
    print("训练曲线已保存：")
    print(loss_save_path)
    print(pa_save_path)
    print(miou_save_path)
    print("=" * 80)


def main():
    log_path = PROJECT_ROOT / "outputs" / "logs" / "unet_baseline_train_log.txt"
    save_dir = PROJECT_ROOT / "outputs" / "figures" / "unet_baseline"

    plot_training_log(
        log_path=log_path,
        save_dir=save_dir,
        model_name="unet_baseline",
    )


if __name__ == "__main__":
    main()