# 无人机图像语义分割

基于 12 类 UAV/CamVid-style 道路场景数据，对比 U-Net、CBAM、ASPP、Weighted CE、DeepLabV3、Swin-UNet、TransUNet 和 SegFormer 路线。

项目保留了训练/评估源码、数据集、最终指标、训练日志、报告图和课堂汇报材料。模型权重与批量预测图可由源码重新生成，因此未提交到 GitHub。

## 实验结果

| 模型 | 初始化 | Mean IoU |
|---|---|---:|
| SegFormer-B1 | 预训练 | **81.74%** |
| DeepLabV3-ResNet50 | ImageNet 预训练 | 70.81% |
| Swin-UNet | ImageNet 预训练 | 69.17% |
| U-Net Baseline | 随机初始化 | 60.02% |
| TransUNet | 随机初始化 | 58.90% |
| U-Net + ASPP + WCE | 随机初始化 | 58.74% |
| U-Net + ASPP | 随机初始化 | 56.80% |
| U-Net + Baseline WCE | 随机初始化 | 55.93% |
| U-Net + CBAM + ASPP | 随机初始化 | 55.74% |
| U-Net + CBAM | 随机初始化 | 54.09% |

本仓库中的本地可复现实验结果位于 `outputs/metrics/`。SegFormer-B1 由小组第三阶段实验提供，本仓库没有其训练源码；结果与结构说明保存在 `2.pptx`、`vit部分.pdf` 和最终讲稿中。

## 数据集

```text
data/raw/
├── train/
│   ├── images/   # 367 张
│   └── labels/
└── test/
    ├── images/   # 101 张
    └── labels/
```

图像统一处理为 `360 × 480`，标签 ID 范围为 `[0, 11]`。

## 安装

建议使用 Python 3.10+ 和支持当前 PyTorch 版本的 CUDA 环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

DeepLabV3 和 Swin-UNet 首次训练时可能自动下载 torchvision 预训练权重。

## 训练与评估

在项目根目录运行：

```powershell
# U-Net Baseline
python train/train.py
python test/test_baseline_augfix.py

# CNN 消融
python train/train_cbam.py
python test/test_cbam.py
python train/train_aspp_fixed_aug.py
python test/test_aspp_fixed_aug.py
python train/train_aspp_weighted_ce.py
python test/test_aspp_weighted_ce.py

# 预训练 CNN
python train/train_deeplab.py
python eval/eval_deeplab.py

# CNN + Transformer
python train/train_transunet.py
python eval/eval_transunet.py
python predict/predict_transunet.py

# 预训练 Transformer
python train/train_swin_unet.py
python eval/eval_swin_unet.py
python predict/predict_swin_unet.py
```

训练生成的权重保存到 `outputs/checkpoints/`，批量预测保存到 `outputs/predictions/`。这两个目录已被 Git 忽略。

## 项目结构

```text
models/       模型定义
train/        训练入口
eval/         DeepLabV3、Swin-UNet、TransUNet 评估入口
test/         U-Net 系列实验评估入口
predict/      可视化预测入口
utils/        数据、损失、指标、训练循环和可视化
tests/        自动化测试
outputs/      小体积日志、指标和最终图表
report/       实验报告、最终讲稿和精选结果图
2.pptx       最终课堂汇报
vit部分.pdf  SegFormer 阶段材料
```

## 测试

```powershell
python -m pytest tests -q -p no:cacheprovider
```

## 结果说明

DeepLabV3 和 Swin-UNet 使用 `data/raw/test` 作为 current evaluation split 进行模型选择和评估，因此报告中不将该结果表述为独立官方 benchmark。

更完整的实验分析见：

- `report/cnn_experiment_report.md`
- `report/uav_segmentation_final_summary_report_2026-06-06.md`
- `report/week14_uav_segmentation_10min_speech_final_2026-06-08.md`
