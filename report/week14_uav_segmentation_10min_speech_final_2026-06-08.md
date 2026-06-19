# 无人机图像语义分割项目 10 分钟正式讲稿

适配来源：

- `D:\uav_segmentation_project\2.pptx`
- `D:\uav_segmentation_project\vit部分.pdf`
- 项目指标：`outputs/metrics/`

建议语速：正常偏稳。  
建议讲法：第 10、12、13、14、15 页是图片页，合并快速讲，不逐页展开。

## 时间安排

| 部分 | 对应页 | 时间 |
|---|---:|---:|
| 开场、目录、背景 | Slide 1-3 | 1.5 min |
| CNN baseline 与改进 | Slide 4-5 | 2 min |
| SegFormer / ViT | Slide 6-7 | 2.5 min |
| DeepLabV3、Swin-UNet、TransUNet | Slide 8-11 | 2 min |
| 违停探索、总结果、总结 | Slide 12-16 | 2 min |

## 正式口播稿

大家好，我们这次汇报的主题是无人机图像语义分割。这个任务的目标是对航拍图像进行像素级分类，也就是把图像中的每一个像素划分到道路、建筑、植被、车辆、行人等语义类别中。相比普通图像分类，语义分割要求更细，因为它不仅要知道图中有什么，还要知道每一类物体具体在哪里。

我们整个项目分为四个阶段：第一阶段是 CNN baseline，第二阶段是 CNN 改进，第三阶段是 ViT 和 SegFormer，第四阶段是 CNN 与 Transformer 融合，也就是 TransUNet。整个实验覆盖 9 个模型、4 个阶段，数据集包含 367 张训练图像、101 张测试图像和 12 个语义类别。最终最优结果来自 SegFormer-B1，mIoU 达到 81.74%。

先看任务背景。无人机航拍图像的难点主要有四个。第一是样本少，训练集只有 367 张，而且场景相似度较高，有效信息量更少。第二是类别长尾，建筑、道路、植被占据大量像素，而车辆、人物、水体这类小目标或稀有类别像素很少。第三是尺度差异明显，建筑和道路是大区域，车辆和人物在低分辨率图像里可能只有几十个像素。第四是边界模糊，比如水体、阴影和路面在颜色和纹理上容易混淆。因此我们把 Mean IoU 作为核心指标，因为它比 Pixel Accuracy 更能反映各类别，尤其是小类别的真实表现。

CNN 阶段首先建立 U-Net baseline。U-Net 是经典的 encoder-decoder 分割结构，编码器负责下采样提取语义，解码器负责上采样恢复空间分辨率，中间通过 skip connection 把浅层边缘细节传回解码端。在工程实现中，我们修复了训练集和验证集共用 Dataset 对象的问题，确保训练集开启增强，验证和测试阶段保持确定性。修复后 U-Net baseline 训练 50 个 epoch，测试 mIoU 达到 60.02%，Pixel Accuracy 为 89.20%。这个结果成为后续所有实验的参照基线。

第二阶段我们尝试了 CNN 改进，包括 CBAM、ASPP 和 Weighted Cross Entropy。CBAM 通过通道注意力和空间注意力让模型关注更重要的区域；ASPP 用不同膨胀率的空洞卷积捕获多尺度上下文；Weighted CE 通过类别频率加权缓解长尾问题。但实验结果说明，模块堆叠并不一定带来提升。CBAM 的 mIoU 是 54.09%，ASPP 是 56.80%，ASPP 加 WCE 是 58.74%，都低于 U-Net baseline 的 60.02%。这说明在 367 张图像的小样本条件下，复杂模块会增加泛化压力，从零训练时简单稳定的 U-Net 反而最可靠。

接下来是 ViT 阶段，重点是 SegFormer。SegFormer 的第一个核心设计是层次化 Transformer 编码器。它分四个 stage 逐步降采样，形成类似特征金字塔的多尺度表达：浅层保留空间细节，深层提取全局语义。第二个设计是 Efficient Self-Attention。传统自注意力复杂度是 O(N²)，而 SegFormer 对 K 和 V 做序列缩减，把复杂度降到 O(N²/R)，这样既保留全局建模能力，又控制计算量。第三个设计是 Mix-FFN，它在前馈网络中加入 3×3 深度可分离卷积，用局部卷积和 zero-padding 隐式表达位置信息，不需要固定绝对位置编码。最后，训练端使用 CE 加 Dice 的 HybridLoss，Dice 对小类别更敏感，可以缓解车辆、人物等稀疏类别被大面积背景淹没的问题。

SegFormer-B1 的结果是全流程最优。根据 PPT 和 PDF 结果，它在第 19 轮达到最佳性能，mIoU 为 81.74%，Pixel Accuracy 为 95.79%。从逐类表现看，建筑、道路、植被这类大面积连续类别表现最好，因为 Transformer 可以跨空间聚合同类像素，使掩码更加连续。车辆和人物这类小目标也明显优于 CNN baseline，说明高分辨率浅层 token 和 Dice Loss 对小目标有效。但水体仍然是短板，IoU 大约在 0.22 到 0.36 之间，主要原因是水体样本少，并且低分辨率下容易和阴影、路面混淆。

为了进一步验证预训练的作用，我们还做了 DeepLabV3 和 Swin-UNet 对照。DeepLabV3-ResNet50 使用 ImageNet 预训练 ResNet50 backbone 和 ASPP 模块，本地评估 mIoU 为 70.81%，明显高于 U-Net baseline，也远高于从零训练的 U-Net + ASPP。这说明 ASPP 本身不是关键，预训练表征才是性能提升的主要来源。

Pretrained Swin-UNet 使用 ImageNet 预训练的 Swin-T 作为 encoder，通过 window attention 和 shifted window 建模局部与跨窗口关系，解码端使用多尺度融合。它的 mIoU 是 69.17%，Pixel Accuracy 是 93.31%，同样明显高于随机初始化的 TransUNet。这个结果说明 Transformer 并不是无效，而是小样本条件下必须依赖预训练，否则很难从 367 张图像中学到稳定的视觉表征。

第四阶段我们尝试 TransUNet，也就是 CNN 与 Transformer 融合。它保留 U-Net 的 encoder-decoder 框架，在 bottleneck 位置把 CNN feature map 转换为 token 序列，送入 Transformer encoder 做全局建模，再恢复为特征图进入解码器。设计目标是结合 CNN 的局部归纳偏置和 Transformer 的全局上下文能力。但最终 TransUNet 的 mIoU 是 58.90%，低于 U-Net baseline 的 60.02%。原因主要是 Transformer 随机初始化、数据量不足，而且 Transformer 只在最深层参与建模，浅层细节仍主要由 CNN 决定。所以 TransUNet 的意义更像对照实验：融合结构本身不保证提升，Transformer 要真正发挥作用仍然需要预训练或更多数据。

关于 PPT 后面几页的违停候选探索，我这里需要特别说明。我们曾尝试在语义分割基础上做疑似违停检测，但“识别出车辆”不等于“判定违停”。违停本质上是车辆与道路、人行道、绿化带、出入口等空间实体之间的拓扑关系判断。早期规则只看车辆周围非道路上下文比例，后来人工复核发现证据不足。进一步检查车辆底部支撑区域后，12 张候选全部是 `reject_not_confirmed`，也就是不能确认真实违停。因此这一部分更适合讲成“疑似违停候选探索和局限性分析”，不能说已经准确识别了违停。

最后总结全模型结果。整体最清楚的结论是：在小样本无人机语义分割中，预训练表征比单纯堆叠结构更重要。SegFormer-B1 最优，mIoU 为 81.74%；本地补充实验中，DeepLabV3-ResNet50 pretrained 为 70.81%，Pretrained Swin-UNet 为 69.17%；从零训练模型里，U-Net baseline 最稳定，为 60.02%；CBAM、ASPP、WCE 和随机初始化 TransUNet 都没有稳定超过它。

因此，我们的核心结论有三点。第一，CNN baseline 是必要参照，不能只看复杂模型自己的结果。第二，Transformer 的全局建模能力有价值，但小样本场景下必须依赖预训练。第三，后续如果继续优化，应优先补充低 IoU 类别数据，改进 CE + Dice 等损失组合，并为违停识别引入更明确的道路边界、停车区域或交通规则标注。以上就是我们的汇报，谢谢大家。

## 备用 Q&A

**Q1：为什么 SegFormer 最好？**  
因为它同时具备层次化多尺度 Transformer、高效自注意力、Mix-FFN 隐式位置编码，以及 CE + Dice 混合损失。

**Q2：为什么 TransUNet 没超过 U-Net？**  
因为 Transformer 是随机初始化，数据只有 367 张，并且 Transformer 只在 bottleneck 参与建模，收益不足以抵消复杂度。

**Q3：为什么 DeepLabV3 和 Swin-UNet 更好？**  
因为它们有 ImageNet 预训练 backbone，预训练提供了稳定的边缘、纹理和语义表征。

**Q4：违停图能不能作为准确结果展示？**  
不能。最新复核中 12 张候选全部是 `reject_not_confirmed`，只能作为疑似候选探索和误判分析。
