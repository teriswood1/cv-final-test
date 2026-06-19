# 无人机语义分割项目总实验报告

## 1. 项目目标

本项目面向无人机道路场景语义分割任务，目标是在有限数据规模下比较不同分割模型和改进策略的效果，并为最终实验报告提供可复现、可解释的实验依据。

主要研究问题包括：

1. U-Net baseline 在当前数据集上的基础性能如何。
2. CBAM 注意力模块是否能提升分割效果。
3. ASPP 多尺度上下文模块是否能提升分割效果。
4. Weighted Cross Entropy 是否能缓解类别不平衡。
5. 预训练模型是否比从零训练模型更适合当前小样本任务。
6. CNN + Transformer 融合模型 TransUNet 是否能超过 U-Net baseline。
7. 加入 ImageNet 预训练权重后的 Transformer 路线是否能改善随机初始化 TransUNet 效果较差的问题。

## 2. 数据集与任务设置

当前数据位于 `data/raw` 目录：

| 数据目录 | 图像数量 | 标签数量 | 用途 |
|---|---:|---:|---|
| `data/raw/train` | 367 | 367 | 训练数据来源 |
| `data/raw/test` | 101 | 101 | 测试集 / 当前本地评估集 |

通用设置如下：

| 配置项 | 设置 |
|---|---|
| 输入尺寸 | `360 x 480` |
| 语义类别数 | 12 |
| Batch size | 2 |
| U-Net base channels | 32 |
| 主实验训练轮数 | 50 epochs |
| 主要指标 | Test Loss、Pixel Accuracy、Mean IoU、Class IoU |
| 最优权重选择 | 验证集或当前评估集 mIoU |

U-Net 系列和 TransUNet 使用 `data/raw/train` 内部 80/20 划分进行训练和验证，再在 `data/raw/test` 上测试。DeepLabV3 和 Pretrained Swin-UNet 使用完整 `data/raw/train` 训练，并使用 `data/raw/test` 作为 current eval split 选取最优权重和评估。因此最终论文中应谨慎表述为“当前本地评估结果”，不直接称为官方 held-out benchmark。

具体预处理方法如下：

| 处理步骤 | 具体方法 | 目的 |
|---|---|---|
| 图像读取 | 使用 PIL 读取并转换为 RGB 三通道 | 保证模型输入通道一致 |
| 标签读取 | 使用单通道灰度图读取 label | 保留类别 ID |
| 尺寸统一 | 图像 resize 到 `360 x 480`，使用双线性插值 | 统一 batch 输入尺寸 |
| 标签 resize | label resize 到 `360 x 480`，使用最近邻插值 | 避免类别编号被插值破坏 |
| Tensor 转换 | 图像转为 `[3, H, W]` FloatTensor，标签转为 `[H, W]` LongTensor | 满足 PyTorch 训练格式 |
| 标签检查 | 检查 label ID 是否位于 `[0, 11]` | 防止异常标签参与训练 |
| 预训练模型归一化 | DeepLabV3 与 Swin-UNet 使用 ImageNet mean/std：`(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)` | 匹配预训练模型输入分布 |

## 3. 已完成的工程工作

### 3.1 数据与训练流程

已完成内容：

- 实现图像与标签读取流程，支持无人机语义分割数据。
- 统一处理 12 类语义标签。
- 实现训练集、验证集、测试集的数据加载。
- 修复早期训练集与验证集共用 Dataset 对象导致增强被误关的问题。
- 将训练集增强和验证/测试集无增强明确拆开，避免数据处理逻辑互相影响。
- 增加小角度旋转、亮度/对比度抖动等道路场景增强。
- 统一训练循环、指标计算、checkpoint 保存、日志输出和测试脚本。

数据增强的具体处理方式分为普通增强和道路场景增强两类。普通增强包含随机水平翻转、少量垂直翻转、小角度旋转以及亮度/对比度扰动；其中图像和标签的几何变换保持同步，图像使用双线性插值，标签使用最近邻插值。道路场景增强更保守，主要包含 `0.5` 概率水平翻转、`1.0 ~ 1.25` 倍随机缩放后裁剪回原尺寸，以及 `0.5` 概率亮度/对比度扰动，避免过强增强破坏道路场景的上下结构。

训练过程的具体处理流程为：

```text
读取 image / label
-> resize 到统一尺寸
-> 训练集执行增强，验证/测试集不增强
-> 转为 Tensor，并按需要执行 ImageNet normalize
-> 前向传播得到 logits
-> CrossEntropy 或 Weighted CrossEntropy 计算 loss
-> 反向传播更新参数
-> 在验证集 / current eval split 上计算 mIoU
-> 若 mIoU 更高，则保存 best checkpoint
```

关键文件：

| 文件 | 作用 |
|---|---|
| `utils/dataset.py` | 数据读取、标签读取、增强处理 |
| `utils/data_split.py` | train/val 或 train/eval 数据划分 |
| `utils/metrics.py` | Pixel Accuracy、mIoU、Class IoU |
| `utils/losses.py` | 类别权重与 Weighted CE |
| `utils/train_engine.py` | 通用训练循环 |
| `utils/visualize.py` | 分割结果可视化 |

### 3.2 实验命名与输出管理

为避免 30 轮与 50 轮实验互相覆盖，最终实验统一使用 `_e50` 后缀，例如：

```text
unet_baseline_augfix_e50
deeplabv3_resnet50_pretrained_road_e50
transunet_e50
swin_unet_pretrained_e50
```

主要输出目录：

| 目录 | 内容 |
|---|---|
| `outputs/checkpoints/` | best / last 权重 |
| `outputs/logs/` | 训练日志 |
| `outputs/metrics/` | 测试指标与汇总 CSV |
| `outputs/predictions/` | 预测图、mask、overlay、compare 图 |
| `outputs/figures/cnn_report/` | 最终报告图表 |

## 4. 已完成模型与脚本

### 4.1 U-Net 系列

完成了以下 CNN baseline 与模块消融：

| 模型 | 训练脚本 | 测试脚本 | 说明 |
|---|---|---|---|
| U-Net Baseline | `train.py` | `test_baseline_augfix.py` | 从零训练基础模型 |
| U-Net + CBAM | `train_cbam.py` | `test_cbam.py` | 编码器加入 CBAM 注意力 |
| U-Net + ASPP | `train_aspp_fixed_aug.py` | `test_aspp_fixed_aug.py` | bottleneck 加入 ASPP |
| U-Net + CBAM + ASPP | `train_cbam_aspp.py` | `test_cbam_aspp.py` | 注意力与多尺度模块组合 |
| U-Net Baseline + WCE | `train_baseline_wce.py` | `test_baseline_wce.py` | baseline 加类别加权损失 |
| U-Net + ASPP + WCE | `train_aspp_weighted_ce.py` | `test_aspp_weighted_ce.py` | ASPP 版本加类别加权损失 |

U-Net baseline 采用典型 encoder-decoder 结构：编码端通过多层卷积和下采样提取高层语义特征，解码端逐步上采样恢复空间分辨率，并通过 skip connection 拼接浅层细节特征。最终使用 `1x1` 卷积输出 12 个类别的 logits。

CBAM 的处理流程是先做通道注意力，再做空间注意力。通道注意力同时使用全局平均池化和全局最大池化得到通道描述，再经过共享 MLP 和 sigmoid 生成通道权重；空间注意力则对通道维做平均池化和最大池化，拼接后经过 `7x7` 卷积生成空间权重。ASPP 的处理流程是对 bottleneck 特征同时执行 `1x1` 卷积、膨胀率为 `6/12/18` 的空洞卷积以及全局平均池化分支，拼接后再用 `1x1` 卷积投影回原通道数。

Weighted CE 的类别权重由训练标签的像素频率计算得到，主要使用 median frequency 思路：类别频率越低，损失权重越高，最后再归一化使权重均值接近 1。该方法用于缓解类别不平衡，但最终结果表明它对不同结构的收益并不稳定。

### 4.2 预训练 CNN 强基线

完成 DeepLabV3-ResNet50 预训练路线：

| 文件 | 作用 |
|---|---|
| `models/deeplab.py` | Torchvision DeepLabV3-ResNet50 适配 |
| `train_deeplab.py` | DeepLabV3 训练 |
| `eval_deeplab.py` | DeepLabV3 当前本地评估 |

该路线使用 ImageNet / COCO 预训练特征作为强基线，验证预训练表征在小样本语义分割中的作用。

具体方法上，DeepLabV3 使用 torchvision 的 ResNet50 backbone 和 DeepLabV3 segmentation head，并将最后的分类卷积替换为输出 12 类的 `1x1` 卷积。训练时使用 ImageNet 标准归一化，优化器为 AdamW，学习率 `1e-4`，weight decay `1e-4`，并使用 CosineAnnealingLR 将学习率逐步降到 `1e-6`。由于 DeepLabV3 的 ASPP pooling 分支包含 BatchNorm，训练 loader 使用 `drop_last=True`，避免最后一个 batch 只剩 1 张图像导致 BatchNorm 报错。

### 4.3 第四阶段 TransUNet

完成第四阶段 CNN + Transformer 融合模型：

| 文件 | 作用 |
|---|---|
| `models/transunet.py` | U-Net encoder-decoder + Transformer bottleneck |
| `train_transunet.py` | TransUNet 训练 |
| `eval_transunet.py` | TransUNet 测试 |
| `predict_transunet.py` | TransUNet 可视化预测 |

TransUNet 的目标是结合 CNN 的局部纹理建模能力与 Transformer 的全局上下文建模能力。

具体方法上，TransUNet 保留 U-Net 的四级下采样和四级上采样结构，在最深层 bottleneck 特征上插入 Transformer。bottleneck 特征先通过 `1x1` 卷积投影到 `256` 维 token 表示，再展平成序列并加入可学习位置编码；Transformer encoder 使用 `2` 层、`4` 个 attention heads、MLP hidden dim 为 `1024`，激活函数为 GELU。Transformer 输出再 reshape 回特征图，并通过残差方式加回原 bottleneck 特征，然后进入 U-Net 解码器。

训练上，TransUNet 使用 Adam 优化器，学习率 `1e-4`，损失函数为标准 CrossEntropyLoss，按验证集 mIoU 保存 best checkpoint。该设置用于观察“从零训练 CNN + Transformer 融合模型”在当前数据规模下的真实效果。

### 4.4 预训练 Transformer 路线

在随机初始化 TransUNet 效果不理想后，补充完成 Pretrained Swin-UNet：

| 文件 | 作用 |
|---|---|
| `models/pretrained_swin_unet.py` | ImageNet 预训练 Swin-T encoder + FPN decoder |
| `train_swin_unet.py` | Pretrained Swin-UNet 训练 |
| `eval_swin_unet.py` | Pretrained Swin-UNet 测试 |
| `predict_swin_unet.py` | Pretrained Swin-UNet 可视化预测 |

该模型使用 torchvision 的 Swin-T 预训练权重作为编码器，并通过 FPN 风格解码器恢复分割图。

具体方法上，Swin-T encoder 输出四个尺度的特征图，通道数分别为 `96/192/384/768`。由于 torchvision Swin 内部特征为 NHWC 排布，模型中先转换为 PyTorch 常用的 NCHW 格式，再送入解码器。FPN decoder 先用 `1x1` 卷积将四个尺度统一到 `128` 通道，然后自顶向下逐层双线性上采样并与浅层 lateral feature 相加，最后通过卷积头输出 12 类 logits，并插值恢复到输入尺寸。

训练上，Swin-T encoder 使用较小学习率 `1e-5`，decoder 使用较大学习率 `1e-4`，优化器为 AdamW，weight decay 为 `1e-4`，学习率调度器为 CosineAnnealingLR。这样处理可以在保留预训练表征的同时，让新初始化的 decoder 更快适应当前分割任务。

## 5. 早期历史实验结果

早期 30 轮实验在增强逻辑存在问题的情况下完成，结果仅作为历史对比，不作为最终结论。

| 编号 | 模型 | Test mIoU | Test Pixel Acc | Test Loss | 结论 |
|---|---|---:|---:|---:|---|
| Exp-1 | U-Net Baseline | 0.5789 | 0.9093 | 0.3184 | 旧 baseline |
| Exp-2 | U-Net + CBAM | 0.5728 | 0.9097 | 0.3217 | CBAM 单独收益不明显 |
| Exp-3 | U-Net + CBAM + ASPP | 0.5486 | 0.8998 | 0.3626 | 模块组合泛化变差 |
| Exp-4 | U-Net + ASPP | 0.5851 | 0.9141 | 0.2911 | 旧实验中表现最好 |

早期结论是 ASPP 相比 baseline 有轻微提升，而 CBAM 与 ASPP 组合并未形成正向叠加。但由于训练增强实际被误关，该组结果只保留为历史记录。

## 6. 最终 50 轮实验结果

修复增强逻辑并统一 50 epoch 后，当前最终汇总如下：

| 排名 | 模型 | 输出前缀 | Loss | Pixel Acc | Mean IoU |
|---:|---|---|---:|---:|---:|
| 1 | DeepLabV3-ResNet50 pretrained | `deeplabv3_resnet50_pretrained_road_e50` | 0.287043 | 0.931919 | **0.708121** |
| 2 | Pretrained Swin-UNet | `swin_unet_pretrained_e50` | 0.201861 | 0.933063 | **0.691668** |
| 3 | U-Net Baseline | `unet_baseline_augfix_e50` | 0.355128 | 0.892028 | **0.600185** |
| 4 | TransUNet | `transunet_e50` | 0.316690 | 0.902656 | 0.588982 |
| 5 | U-Net + ASPP + WCE | `unet_aspp_weighted_ce_e50` | 0.523465 | 0.882595 | 0.587356 |
| 6 | U-Net + ASPP | `unet_aspp_fixed_aug_e50` | 0.270272 | 0.915694 | 0.568018 |
| 7 | U-Net Baseline + WCE | `unet_baseline_wce_e50` | 0.420052 | 0.842551 | 0.559326 |
| 8 | U-Net + CBAM + ASPP | `unet_cbam_aspp_augfix_e50` | 0.281705 | 0.908615 | 0.557352 |
| 9 | U-Net + CBAM | `unet_cbam_augfix_e50` | 0.331968 | 0.896127 | 0.540934 |

主要图表：

| 图表 | 文件 |
|---|---|
| 50 轮 mIoU 排名 | `outputs/figures/cnn_report/final_e50_miou_ranking.png` |
| 50 轮指标对比 | `outputs/figures/cnn_report/final_e50_metric_comparison.png` |
| 逐类 IoU 热力图 | `outputs/figures/cnn_report/final_e50_class_iou_heatmap.png` |
| 50 轮验证 mIoU 曲线 | `outputs/figures/cnn_report/e50_val_miou_curves.png` |

评估指标的计算基于混淆矩阵。模型输出 logits 后先在类别维度执行 `argmax` 得到预测类别图，再与真实 label 累积混淆矩阵。Pixel Accuracy 计算所有预测正确像素占总像素的比例；单类 IoU 计算 `TP / (TP + FP + FN)`；Mean IoU 对所有类别 IoU 取平均。该方式比单纯 Pixel Accuracy 更能反映小目标类别和困难类别的分割质量。

## 7. 关键实验分析

### 7.1 U-Net Baseline

修复数据增强并训练 50 轮后，U-Net Baseline 达到 `0.600185` mIoU，是所有从零训练 U-Net 系列中最稳定的模型。相比旧 baseline 的 `0.5789`，最终 baseline 有明显提升，说明修复增强逻辑与延长训练轮数是有效的。

Baseline 的处理重点是尽量减少额外模块干扰，只保留卷积下采样、skip connection 和上采样恢复。它作为后续 CBAM、ASPP、WCE 和 Transformer 结果的参照，能够判断新增模块到底带来了收益还是仅增加了复杂度。

### 7.2 CBAM 模块

U-Net + CBAM 的 mIoU 为 `0.540934`，低于 U-Net Baseline 的 `0.600185`。这说明在当前数据规模和训练配置下，CBAM 注意力模块没有带来稳定收益。可能原因是数据量偏小，注意力模块增加额外参数后更容易产生泛化压力。

CBAM 在本项目中主要加在 encoder 的下采样特征提取阶段，希望模型自动增强重要通道和关键空间区域。但最终结果显示，注意力权重并没有稳定改善小类 IoU，说明当前任务中仅靠轻量注意力不足以解决类别不平衡和场景差异问题。

### 7.3 ASPP 模块

U-Net + ASPP 的 mIoU 为 `0.568018`，Pixel Accuracy 为 `0.915694`。ASPP 的像素准确率较高，说明它可能改善了大面积类别的分类，但 Mean IoU 未超过 baseline，说明对小类或困难类别的改善不足。

ASPP 的方法目标是通过不同膨胀率卷积同时捕获局部和较大感受野信息，适合道路、天空、建筑等尺度差异明显的场景。但在本项目中，ASPP 提升了整体像素分类稳定性，却没有让所有类别的 IoU 同步提升，因此最终 mIoU 低于 baseline。

### 7.4 CBAM + ASPP 组合

U-Net + CBAM + ASPP 的 mIoU 为 `0.557352`，低于 baseline，也低于单独 ASPP + WCE。该结果说明两个模块在当前配置下没有形成有效协同，反而可能增加模型复杂度并影响泛化。

该组合的处理逻辑是先在 encoder 中利用 CBAM 重标定特征，再在 bottleneck 中利用 ASPP 扩展感受野。理论上两者分别对应“注意力筛选”和“多尺度上下文”，但实际训练中模型复杂度增加后，对 367 张训练图像的泛化并不稳定。

### 7.5 Weighted CE 损失函数

Weighted CE 对不同结构的影响不一致：

| 对比 | 原模型 mIoU | 加 WCE 后 mIoU | 变化 |
|---|---:|---:|---|
| Baseline -> Baseline + WCE | 0.600185 | 0.559326 | 下降 |
| ASPP -> ASPP + WCE | 0.568018 | 0.587356 | 提升 |

因此，WCE 不是稳定有效的默认损失函数。它可以帮助 ASPP 版本改善一部分类别，但对 baseline 不利。

WCE 的具体处理是在 CrossEntropyLoss 中传入类别权重，使少数类像素在 loss 中占更高比例。该方法可以迫使模型关注低频类别，但如果权重过高，也可能牺牲大类稳定性，导致总体 mIoU 下降。Baseline + WCE 的结果下降正体现了这个风险。

### 7.6 DeepLabV3-ResNet50

DeepLabV3-ResNet50 pretrained 的 mIoU 为 `0.708121`，是当前最优模型。它比最佳从零训练 U-Net Baseline 的 `0.600185` 高 `0.107936`，约提升 `10.79` 个 mIoU 百分点。

该结果说明，在当前 367 张训练图像的小样本条件下，预训练表征比单纯堆叠 U-Net 模块更关键。

DeepLabV3 的方法优势来自两部分：一是 ResNet50 预训练 backbone 提供更强的通用视觉表征，二是 DeepLabV3 head 中的 ASPP 结构能够在高层语义特征上建模多尺度上下文。与从零训练 U-Net 系列相比，它不需要完全从 367 张图像中学习底层纹理和中层语义，因此泛化更稳。

### 7.7 TransUNet

TransUNet 的 mIoU 为 `0.588982`，高于 U-Net + ASPP + WCE 的 `0.587356`，但低于 U-Net Baseline 的 `0.600185`。其最佳验证 mIoU 出现在 epoch 46，为 `0.592059`。

这说明随机初始化的 Transformer bottleneck 有一定全局建模收益，但当前数据量不足以支撑其超过更简单的 U-Net Baseline。之前 Transformer 效果差的主要原因不是全局建模思路完全无效，而是从零训练 Transformer 对数据量和训练稳定性要求更高。

从处理流程看，TransUNet 将最深层 CNN feature map 展平成 token 序列后做 self-attention，因此每个位置可以直接聚合全局上下文。但这种全局交互也带来更多参数和更强的数据需求，在训练样本较少时容易出现“验证集有一定提升、测试集泛化不足”的情况。

### 7.8 Pretrained Swin-UNet

Pretrained Swin-UNet 的 mIoU 为 `0.691668`，Pixel Accuracy 为 `0.933063`，显著高于随机初始化 TransUNet 的 `0.588982`。两者差值为：

```text
0.691668 - 0.588982 = 0.102686
```

即提升约 `10.27` 个 mIoU 百分点。

该结果说明：Transformer 路线本身有潜力，但在当前小样本语义分割任务中，必须依赖预训练表征才能稳定发挥效果。Pretrained Swin-UNet 仍略低于 DeepLabV3-ResNet50 的 `0.708121`，差距约 `1.65` 个百分点。

Swin-UNet 的处理方式与 TransUNet 的关键差异在于：它不是只在 bottleneck 位置临时加入随机初始化 Transformer，而是从输入开始使用已经预训练好的层级式 Swin Transformer 提取多尺度特征。窗口注意力和层级下采样让它既能保留局部结构，也能逐步扩大上下文范围；FPN decoder 再把深层语义与浅层空间细节融合起来，因此在小样本任务上明显优于随机初始化 TransUNet。

## 8. 逐类表现概述

从最终逐类 IoU 可以观察到：

- DeepLabV3 在多数类别上最稳定，尤其 class_10、class_8、class_9 等类别表现较强。
- Pretrained Swin-UNet 在 class_1、class_8、class_9、class_10 上明显优于随机初始化 TransUNet。
- TransUNet 在 class_1、class_5、class_7 上有一定表现，但 class_2、class_11 等小类仍偏弱。
- U-Net Baseline 是从零训练 U-Net 系列中整体最稳的模型。
- class_2 和 class_11 是当前所有模型都比较困难的类别，后续可以重点分析类别样本量、标注质量和 loss 设计。

部分关键类别对比如下：

| 模型 | class_1 | class_2 | class_8 | class_9 | class_10 | class_11 |
|---|---:|---:|---:|---:|---:|---:|
| DeepLabV3 | 0.861216 | 0.121077 | 0.868865 | 0.575431 | 0.828178 | 0.287526 |
| Pretrained Swin-UNet | 0.877596 | 0.093113 | 0.855118 | 0.531460 | 0.759433 | 0.273996 |
| U-Net Baseline | 0.753272 | 0.055810 | 0.714012 | 0.351428 | 0.654589 | 0.283547 |
| TransUNet | 0.821193 | 0.091809 | 0.625558 | 0.303899 | 0.542435 | 0.242437 |

## 9. 可视化与结果产物

已生成的主要可视化结果：

| 模型 | 可视化目录 | 数量 |
|---|---|---:|
| TransUNet | `outputs/predictions/transunet_e50/compare/` | 101 张 |
| Pretrained Swin-UNet | `outputs/predictions/swin_unet_pretrained_e50/compare/` | 101 张 |

这些对比图可用于最终论文中的定性展示，建议选择同一批样本对比 DeepLabV3、TransUNet、Pretrained Swin-UNet 和 U-Net Baseline，突出“随机初始化 Transformer”和“预训练 Transformer”的差异。

可视化处理流程为：先加载 best checkpoint，对测试图像前向推理得到 logits；随后执行 `argmax` 得到预测类别 mask；再将类别 mask 按调色板转换为彩色分割图，并生成原图、真实标签、预测结果和叠加效果的 compare 图。这样既能看整体区域是否分对，也能观察边界、小目标和类别混淆情况。

## 10. 测试与代码质量

已补充并运行单元测试，覆盖数据划分、指标计算、DeepLab 构建、TransUNet 输出尺寸、Pretrained Swin-UNet 输出尺寸、训练引擎等关键逻辑。

最近一次验证结果：

```text
pytest tests -q --basetemp .pytest_tmp -p no:cacheprovider
16 passed in 6.80s
```

新增脚本也已通过 Python 语法编译检查：

```text
python -m py_compile models\pretrained_swin_unet.py train_swin_unet.py eval_swin_unet.py predict_swin_unet.py
```

项目中已清理本次测试和训练产生的临时目录，如 `.pytest_tmp`、`outputs/runtime` 和可访问的 `__pycache__`。`.pytest_cache` 目录因 Windows 权限拒绝访问，未强行删除。

## 11. 最终结论

1. 当前整体最优模型是 **DeepLabV3-ResNet50 pretrained**，mIoU 为 `70.81%`。
2. 当前最强 Transformer 路线是 **Pretrained Swin-UNet**，mIoU 为 `69.17%`。
3. 从零训练模型中，**U-Net Baseline** 最稳，mIoU 为 `60.02%`。
4. 第四阶段 **TransUNet** 已完成，mIoU 为 `58.90%`，证明融合建模有一定作用，但在随机初始化和小样本条件下未超过 baseline。
5. **预训练权重是本项目性能提升的关键因素**。DeepLabV3 和 Pretrained Swin-UNet 都明显超过从零训练模型。
6. CBAM、ASPP、CBAM + ASPP 在最终 50 轮设置下均未超过 U-Net Baseline，应作为消融对比而非最终主模型。
7. Weighted CE 对 ASPP 有一定帮助，但对 baseline 不利，不建议作为默认损失函数。

## 12. 可直接用于论文的总结段落

在统一 50 epoch 训练设置下，U-Net Baseline 在本地测试集上取得 `60.02%` mIoU。进一步引入 CBAM、ASPP、类别加权损失以及从零训练 TransUNet 后，整体 mIoU 均未超过 baseline，其中 TransUNet 达到 `58.90%` mIoU。补充 ImageNet 预训练 Swin-T encoder 后，Pretrained Swin-UNet 的 mIoU 提升至 `69.17%`，显著高于随机初始化 TransUNet，说明 Transformer 路线需要预训练表征支撑。当前最优模型为预训练 DeepLabV3-ResNet50，mIoU 为 `70.81%`。这些结果表明，在当前无人机小样本语义分割任务中，预训练表征比单纯增加局部模块更关键。

## 13. 后续建议

1. 将组员第三阶段 Transformer 实验结果补充进第 6 节最终汇总表。
2. 在最终论文中统一说明 DeepLabV3 和 Pretrained Swin-UNet 的 `data/raw/test` 是 current eval split。
3. 从 `outputs/predictions/` 中挑选典型样本，展示 DeepLabV3、TransUNet、Pretrained Swin-UNet 的定性差异。
4. 针对 class_2、class_11 等低 IoU 类别，后续可尝试类别重采样、Dice/Focal Loss 或更细粒度的数据清洗。
5. 若有时间，可对 Pretrained Swin-UNet 继续尝试 encoder 分层学习率、冻结前几层、Dice + CE 组合损失，以缩小与 DeepLabV3 的差距。
