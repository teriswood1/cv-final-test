# Week14 Academic Talk 演讲讲稿：无人机语义分割项目

## 0. 汇报设计原则

根据 Week14 PDF 的要求，这次汇报应当做到：

- 背景和动机约占 20%，重点说明任务为什么重要。
- 方法和核心想法约占 60%，重点讲清楚 CNN、Transformer、TransUNet 的设计逻辑。
- 实验结果约占 20%，用少量图表证明方法被正确评估。
- 每页 PPT 只传递一个主信息，页面上只放关键词和图表，具体解释由口播完成。

本稿按 10 分钟设计，推荐 8 页 PPT。

| 部分 | 时间 | 对应 PPT |
|---|---:|---|
| 开场与任务背景 | 1.5 min | Slide 1-2 |
| CNN baseline 与改进 | 2.0 min | Slide 3-4 |
| ViT / 预训练 Transformer 路线 | 2.0 min | Slide 5 |
| TransUNet 融合建模 | 2.0 min | Slide 6 |
| 结果分析与总结 | 2.0 min | Slide 7-8 |

> 统合提示：第三阶段独立 ViT 实验由组员完成。最终合并 PPT 时，可以把组员的 ViT 模型、指标和可视化图插入 Slide 5；如果需要先展示当前仓库结果，则可使用 Pretrained Swin-UNet 作为预训练 Transformer 路线的补充说明。

## Slide 1. 标题页：UAV Semantic Segmentation

**PPT 主标题**

UAV Semantic Segmentation: From CNN Baselines to Transformer Fusion

**页面 bullet**

- Task: pixel-level semantic labeling for UAV road scenes
- Data: 367 training images, 101 test images, 12 classes
- Focus: CNN, ViT-style Transformer, TransUNet fusion

**口播稿，约 40 秒**

大家好，我们这次汇报的主题是无人机道路场景语义分割。这个任务的目标不是判断一张图里面有没有某个物体，而是要给图像中的每一个像素分配一个语义类别，例如道路、建筑、车辆、行人或者背景区域。它的难点在于，无人机图像通常视角高、尺度变化大，而且不同类别的像素数量非常不均衡，一些小目标类别很容易被大面积类别淹没。

我们这部分工作的主线是：先完成 CNN baseline，再在 CNN 框架内做模块改进，然后引入 Transformer 路线，最后尝试 CNN 和 Transformer 的融合建模。接下来我会重点讲我负责的 CNN 实验、预训练 Transformer 补充实验和第四阶段 TransUNet 融合实验。

## Slide 2. Motivation and Experimental Pipeline

**PPT 主标题**

One question: what actually improves UAV segmentation under limited data?

**页面 bullet**

- U-Net gives a clean CNN baseline
- Attention and multi-scale modules test local improvements
- Pretraining tests whether stronger representation matters

**建议配图**

使用一张流程图：`Data -> U-Net / CBAM / ASPP / DeepLab / Swin / TransUNet -> mIoU + visualization`

可直接放入 PPT 的仓库图：

![50 epoch validation mIoU curves](week14_talk_assets/e50_val_miou_curves.png)

**口播稿，约 50 秒**

我们最核心的问题是：在只有 367 张训练图像的条件下，到底什么方法真正能提升无人机语义分割效果？所以我们的实验不是只跑一个模型，而是按阶段逐步展开。

第一步是 U-Net baseline，用它建立完整的训练、测试和可视化流程。第二步是在 U-Net 内部加入 CBAM 注意力模块、ASPP 多尺度模块和 Weighted CE 损失，观察这些局部改进是否有效。第三步是引入 Transformer 思路，因为 Transformer 擅长全局建模。第四步是 TransUNet，把 CNN 的局部特征和 Transformer 的全局上下文结合起来。

评估上，我们统一关注 Loss、Pixel Accuracy、Mean IoU 和逐类 IoU，其中 Mean IoU 是最重要的指标，因为它比 Pixel Accuracy 更能反映小类别和困难类别的表现。

## Slide 3. CNN Baseline: U-Net as the Starting Point

**PPT 主标题**

U-Net is the most stable from-scratch baseline

**页面 bullet**

- Encoder-decoder with skip connections
- 50 epochs after fixing augmentation
- Final mIoU: 60.02%

**建议配图**

放一张 U-Net 简化结构图，或放 `outputs/figures/cnn_report/e30_vs_e50_miou_comparison.png`

可直接放入 PPT 的仓库图：

![30 vs 50 epoch mIoU comparison](week14_talk_assets/e30_vs_e50_miou_comparison.png)

**口播稿，约 60 秒**

CNN 部分我们首先使用 U-Net 作为 baseline。选择 U-Net 的原因很直接：它是语义分割中非常经典的 encoder-decoder 结构，编码器负责逐步下采样并提取高层语义特征，解码器负责逐步上采样恢复空间分辨率，同时通过 skip connection 把浅层细节信息传回解码端。

在工程上，我们修复了一个比较关键的问题：早期训练集和验证集共用了同一个 Dataset 对象，导致关闭验证集增强时可能同时关闭训练集增强。修复后，训练集和验证集分别构造 Dataset，训练集开启增强，验证集和测试集关闭增强。

修复后 U-Net baseline 训练 50 个 epoch，最终 mIoU 达到 60.02%。这个结果在从零训练的 U-Net 系列里是最稳定的，因此后面所有模块改进都要和它对比，而不是只看单个模型自己的结果。

## Slide 4. CNN Improvements: CBAM, ASPP, and Weighted CE

**PPT 主标题**

More modules did not always mean better segmentation

**页面 bullet**

- CBAM: channel + spatial attention
- ASPP: multi-scale context by dilated convolutions
- WCE: reweights rare classes

**建议配图**

使用 `outputs/figures/cnn_report/final_e50_miou_ranking.png`，只在讲解时强调 U-Net 系列几项。

可直接放入 PPT 的仓库图：

![Final 50 epoch mIoU ranking](week14_talk_assets/final_e50_miou_ranking.png)

**口播稿，约 60 秒**

在第二阶段，我们尝试了三类 CNN 框架内的改进。

第一类是 CBAM 注意力模块。它先做通道注意力，再做空间注意力，希望模型自动关注更重要的通道和图像区域。第二类是 ASPP，也就是 Atrous Spatial Pyramid Pooling。它通过不同膨胀率的空洞卷积同时捕获多个尺度的上下文信息。第三类是 Weighted Cross Entropy，它根据类别像素频率给少数类更高的损失权重，希望缓解类别不平衡。

但实验结果比较有意思：这些模块并没有稳定超过 U-Net baseline。U-Net + CBAM 的 mIoU 是 54.09%，U-Net + ASPP 是 56.80%，ASPP + WCE 提升到 58.74%，但仍低于 baseline 的 60.02%。这说明在我们这个数据规模下，简单增加模块不一定带来更好的泛化，尤其是注意力模块和多尺度模块可能会增加模型复杂度，小数据上更容易不稳定。

这一页想传达的核心结论是：CNN 消融实验不能只证明“我加了模块”，还必须证明这个模块真的提升了最终 mIoU。我们的结果反而说明，U-Net baseline 是从零训练路线中最稳的。

## Slide 5. ViT / Pretrained Transformer Route

**PPT 主标题**

Transformer needs pretraining to work well on this small dataset

**页面 bullet**

- ViT-style models provide global context
- From-scratch Transformer is data-hungry
- Pretrained Swin-UNet mIoU: 69.17%

**建议配图**

放 Pretrained Swin-UNet 简化图：`Swin-T encoder -> multi-scale features -> FPN decoder -> segmentation mask`。最终统合时，组员第三阶段 ViT 结果可以放在右侧小表中。

可直接放入 PPT 的仓库图：

![Pretrained Swin-UNet qualitative result](week14_talk_assets/swin_unet_0016E5_07959_compare.png)

**口播稿，约 2 分钟**

第三阶段的核心问题是：如果从 CNN 框架切换到 Vision Transformer 或 Transformer-style 模型，效果会不会更好？Transformer 的优势在于它可以建模更长距离的依赖关系。对于无人机图像来说，同一个类别可能分布在画面的不同区域，比如道路、建筑或者背景，长距离上下文理论上是有帮助的。

但是 Transformer 也有明显劣势：它通常比 CNN 更依赖大规模数据。如果直接在小数据集上从零训练，模型可能很难学到稳定的底层视觉特征。这一点也解释了为什么我们后面的随机初始化 TransUNet 没有明显超过 U-Net baseline。

为了验证问题到底出在 Transformer 思路本身，还是出在从零训练的数据不足，我们额外补充了一个 Pretrained Swin-UNet。它使用 ImageNet 预训练的 Swin-T 作为 encoder，提取四个尺度的特征，通道数分别是 96、192、384 和 768。然后通过 FPN 风格的 decoder，把不同尺度统一到 128 通道，自顶向下融合，最后输出 12 类分割图。

这个结果非常关键：Pretrained Swin-UNet 的 mIoU 达到 69.17%，明显高于 U-Net baseline 的 60.02%，也明显高于随机初始化 TransUNet 的 58.90%。所以我们的判断是，Transformer 路线不是无效，而是需要预训练表征支撑。对于当前这种 367 张训练图像的小样本任务，预训练权重比单纯更换结构更重要。

统合时，如果组员的第三阶段独立 ViT 实验已经有完整指标，可以在这一页先展示组员结果，然后用 Pretrained Swin-UNet 作为补充对照：同样是 Transformer 思路，预训练之后性能会显著更稳。

## Slide 6. TransUNet: CNN Local Features + Transformer Global Context

**PPT 主标题**

TransUNet fuses local CNN features with global Transformer context

**页面 bullet**

- U-Net encoder-decoder keeps local details
- Transformer bottleneck models global relations
- Final mIoU: 58.90%

**建议配图**

放 TransUNet 简化结构：`U-Net encoder -> Transformer bottleneck -> U-Net decoder`，旁边放一张 `outputs/predictions/transunet_e50/compare/` 中的样例。

可直接放入 PPT 的仓库图：

![TransUNet qualitative result](week14_talk_assets/transunet_0016E5_07959_compare.png)

**口播稿，约 2 分钟**

第四阶段要求我们进行融合建模，也就是把 CNN 的局部建模能力和 Transformer 的全局建模能力结合起来。我们采用的模型是 TransUNet。

具体来说，我们保留 U-Net 的整体 encoder-decoder 框架。前半部分仍然用卷积和下采样提取局部纹理与边缘信息；到 bottleneck 位置后，把最深层 CNN feature map 通过 1x1 卷积投影到 256 维 token 表示，再展平成序列，加入可学习的位置编码，然后送入 Transformer encoder。这里 Transformer 使用 2 层、4 个 attention heads，MLP hidden dim 是 1024。Transformer 输出再 reshape 回特征图，并通过残差方式加回 bottleneck feature，最后进入 U-Net decoder 恢复分割结果。

从设计直觉上讲，TransUNet 希望解决 U-Net 的一个限制：卷积更擅长局部模式，但对远距离区域之间的关系建模不够直接。Transformer bottleneck 可以让图像中不同位置的高层语义特征进行全局交互。

实验结果是，TransUNet 的测试 mIoU 为 58.90%，略高于 U-Net + ASPP + WCE 的 58.74%，但低于 U-Net baseline 的 60.02%。它的最佳验证 mIoU 出现在 epoch 46，为 59.21%。这说明融合建模确实带来了一些全局上下文收益，但在随机初始化、小样本训练的设置下，收益不足以超过更简单稳定的 U-Net baseline。

所以这一阶段的结论不是“TransUNet 失败”，而是更具体：从零训练的 Transformer 融合模块在当前数据规模下不够稳定；如果希望 Transformer 真正发挥优势，需要预训练或更多数据支撑。

## Slide 7. Results: Pretraining is the strongest factor

**PPT 主标题**

The strongest results come from pretrained representations

**页面 bullet**

- DeepLabV3: 70.81% mIoU
- Pretrained Swin-UNet: 69.17% mIoU
- Best from-scratch model: U-Net baseline, 60.02% mIoU

**建议配图**

优先使用 `outputs/figures/cnn_report/final_e50_metric_comparison.png`。如果页面太拥挤，只保留 top 4 表格：

可直接放入 PPT 的仓库图：

![Final 50 epoch metric comparison](week14_talk_assets/final_e50_metric_comparison.png)

![Final 50 epoch class IoU heatmap](week14_talk_assets/final_e50_class_iou_heatmap.png)

| Model | mIoU |
|---|---:|
| DeepLabV3-ResNet50 | 70.81 |
| Pretrained Swin-UNet | 69.17 |
| U-Net Baseline | 60.02 |
| TransUNet | 58.90 |

**口播稿，约 70 秒**

把所有实验放到一起看，最清楚的结论是：预训练表征是性能提升最大的因素。

当前最优模型是 DeepLabV3-ResNet50 pretrained，mIoU 达到 70.81%。第二名是 Pretrained Swin-UNet，mIoU 是 69.17%。它们都明显高于从零训练的 U-Net baseline，也高于随机初始化 TransUNet。

从零训练模型里，U-Net baseline 反而是最稳的，mIoU 为 60.02%。CBAM、ASPP、Weighted CE 和 TransUNet 都没有超过它。这说明在小样本无人机语义分割中，模型结构更复杂不一定更好；如果没有预训练，复杂模块可能带来泛化压力。

另外，逐类 IoU 也支持这个结论。Pretrained Swin-UNet 在 class_8、class_9、class_10 等类别上明显优于 TransUNet。例如 class_9 从 TransUNet 的 0.3039 提升到 Swin-UNet 的 0.5315。这说明预训练 Transformer 不仅提升平均指标，也改善了一些困难类别。

## Slide 8. Conclusion and Takeaways

**PPT 主标题**

Main takeaway: representation matters more than adding modules

**页面 bullet**

- CNN baseline is necessary for fair comparison
- Transformer is useful, but pretraining is critical
- Final recommendation: DeepLabV3 or Pretrained Swin-UNet

**建议配图**

放 2-3 张 qualitative compare 图：U-Net / TransUNet / Pretrained Swin-UNet 或 DeepLabV3 的预测对比。优先使用 `outputs/predictions/swin_unet_pretrained_e50/compare/` 和 `outputs/predictions/transunet_e50/compare/` 中同一测试样例。

可直接放入 PPT 的仓库图：

![TransUNet same-sample qualitative result](week14_talk_assets/transunet_0016E5_07959_compare.png)

![Pretrained Swin-UNet same-sample qualitative result](week14_talk_assets/swin_unet_0016E5_07959_compare.png)

**口播稿，约 70 秒**

最后总结一下我们这部分工作的三个结论。

第一，CNN baseline 非常重要。它不仅是第一阶段的任务要求，也是判断后续模块是否有效的参照。我们的实验表明，修复增强后 U-Net baseline 是从零训练模型中最稳定的。

第二，Transformer 思路是有价值的，但前提是要有足够好的表征。随机初始化 TransUNet 没有超过 U-Net baseline，而 Pretrained Swin-UNet 达到 69.17% mIoU，说明性能差异主要来自预训练特征，而不是 Transformer 本身完全不适合这个任务。

第三，最终推荐模型是 DeepLabV3-ResNet50 或 Pretrained Swin-UNet。DeepLabV3 当前 mIoU 最高，为 70.81%；Swin-UNet 略低一些，为 69.17%，但它证明了预训练 Transformer 路线的潜力。

如果后续继续优化，我会优先考虑两条路：一是围绕低 IoU 类别做数据和损失函数改进，比如 class_2 和 class_11；二是继续微调预训练 Transformer，例如冻结部分 encoder、调整分层学习率，或者尝试 Dice + CE 组合损失。

以上就是我负责部分的汇报内容，后面可以把组员第三阶段独立 ViT 的实验结果合并进 Slide 5 和最终结果表中，形成完整小组版本。

## PPT 制作清单

为满足 Week14 PDF 的课堂汇报要求，制作 PPT 时建议：

1. 每页只保留一个结论式标题，例如“Pretraining is the strongest factor”。
2. 每页 bullet 控制在 3 条以内，具体解释放到口播。
3. 结果页不要放完整 9 行大表，只放 top 4 或改用柱状图。
4. 图表优先使用 `final_e50_miou_ranking.png`、`final_e50_metric_comparison.png` 和模型预测 compare 图。
5. 第三阶段组员结果统合时，统一术语为 “ViT / Transformer route”，避免一会儿叫 ViT、一会儿叫 Transformer、一会儿叫 Swin 导致听众困惑。

已整理到汇报 assets 目录的图片：

| 图片 | 建议用途 |
|---|---|
| `report/week14_talk_assets/e50_val_miou_curves.png` | Slide 2，说明训练与验证过程 |
| `report/week14_talk_assets/e30_vs_e50_miou_comparison.png` | Slide 3，说明 50 epoch 训练必要性 |
| `report/week14_talk_assets/final_e50_miou_ranking.png` | Slide 4，说明 CNN 消融排名 |
| `report/week14_talk_assets/swin_unet_0016E5_07959_compare.png` | Slide 5，说明预训练 Transformer 定性结果 |
| `report/week14_talk_assets/transunet_0016E5_07959_compare.png` | Slide 6，说明 TransUNet 定性结果 |
| `report/week14_talk_assets/final_e50_metric_comparison.png` | Slide 7，说明总指标对比 |
| `report/week14_talk_assets/final_e50_class_iou_heatmap.png` | Slide 7，说明逐类 IoU 差异 |

## 备用 Q&A

**问题 1：为什么 TransUNet 没有超过 U-Net baseline？**  
答：主要原因是当前数据规模较小，TransUNet 中的 Transformer bottleneck 是随机初始化的，对数据量和训练稳定性要求更高。它有一定全局建模收益，但不足以抵消复杂度带来的泛化压力。

**问题 2：为什么 Pretrained Swin-UNet 明显更好？**  
答：因为它不是从零学习视觉特征，而是使用 ImageNet 预训练 Swin-T encoder。预训练特征提供了更稳定的纹理、边缘和语义表达，FPN decoder 再把多尺度特征融合起来，所以小样本场景下表现明显更稳。

**问题 3：为什么 DeepLabV3 最高？**  
答：DeepLabV3 同时具备预训练 ResNet50 backbone 和 ASPP 多尺度上下文模块。它既有强预训练表征，又有成熟的语义分割 head，因此在当前本地评估中 mIoU 最高。

**问题 4：为什么 CBAM、ASPP 没有提升？**  
答：它们增加了模型复杂度，但当前数据只有 367 张训练图像。CBAM 没有稳定改善小类，ASPP 虽然提高了 Pixel Accuracy，但对困难类别 IoU 改善不足，所以最终 mIoU 低于 baseline。

**问题 5：最终论文应该怎么表述测试集？**  
答：DeepLabV3 和 Pretrained Swin-UNet 使用 `data/raw/test` 作为 current eval split，因此论文中建议表述为“当前本地评估结果”，不要直接称为官方 held-out benchmark。
