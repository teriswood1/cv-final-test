# 实验报告：基于 SegFormer 的无人机语义分割

## 1. 实验目的
构建并训练基于 SegFormer 的语义分割模型，实现无人机影像中道路、植被、水体、建筑、车辆、行人等地物的像素级分类。

## 2. 数据集与类别
- 数据组织：`train/images` + `train/labels`，`test/images` + `test/labels`
- 类别数：6
- 类别名称：Road / Vegetation / Water / Building / Vehicle / Person
- 标签处理：非合法标签统一映射为 `ignore_index=255`
- 待补充：训练集/测试集样本数量（原图数量与总 patch 数）

## 3. 模型与方法
- 模型：SegFormer（`nvidia/mit-b1`）
- 输出类别：6
- 损失函数：HybridLoss = CrossEntropyLoss + DiceLoss
  - 交叉熵保证整体分类稳定
  - Dice 强化小目标类别（如车辆、行人）
- 忽略标签：`ignore_index=255`

## 4. 训练设置
- 输入裁剪：随机 512x512 patch
- 训练样本扩增：每张图采样 4 个 patch
- 数据增强：随机旋转 90 度、水平翻转、垂直翻转、颜色扰动（亮度/对比度/饱和度/色相）
- 归一化：ImageNet 均值/方差
- 优化器：AdamW
- 学习率：6e-4
- 权重衰减：0.01
- 学习率调度：CosineAnnealingLR（T_max=50）
- Batch size：4
- Epoch：50
- 随机种子：42
- 设备：自动选择 GPU（如无则 CPU）
- 待补充：硬件型号、显存、单 epoch 时间、总训练时间

## 5. 评价指标
- Pixel Accuracy (PA)
- Mean IoU (mIoU)
- 各类别 IoU

## 6. 实验结果
- 最佳 mIoU 出现在 Epoch 19：
  - mIoU = 0.8448
  - PA = 0.9714
  - 类别 IoU：
    - Road 0.9435
    - Vegetation 0.9414
    - Water 0.3712
    - Building 0.9828
    - Vehicle 0.8929
    - Person 0.9371

- 训练结束（Epoch 50）指标：
  - mIoU = 0.8339
  - PA = 0.9704
  - 类别 IoU：
    - Road 0.9450
    - Vegetation 0.9377
    - Water 0.3122
    - Building 0.9827
    - Vehicle 0.8936
    - Person 0.9320

## 7. 结果分析
- 模型在道路、植被、建筑、车辆、行人等类别上达到较高 IoU（普遍超过 0.88），说明网络对结构明显、纹理清晰的地物具有较强分割能力。
- 水体 IoU 明显偏低（约 0.31-0.37），推测原因包括：
  1) 类别样本占比小，导致训练不足
  2) 水体光谱/纹理与阴影、道路等易混淆
  3) 小尺度或细碎水体难以通过随机裁剪被充分覆盖
- 从训练曲线看，mIoU 在 10-20 epoch 之间达到高峰，后期略有波动，说明已进入收敛区间，可考虑使用早停或保存最佳权重。

## 8. 结论
本实验基于 SegFormer-B1 进行无人机影像语义分割训练，在 6 类场景下取得 mIoU 0.8448 的最佳验证表现，整体精度较高。模型对结构显著类别（道路、建筑、植被）表现优异，但对水体类仍有提升空间。

## 9. 后续改进方向
- 数据层面：增加水体样本或进行类别重加权
- 训练策略：采用类别加权 CE / Focal Loss
- 推理策略：采用滑窗 + 多尺度测试
- 结构优化：尝试更大 backbone（如 SegFormer-B2/B3）

## 10. 待补充信息
- 训练集与测试集的原图数量与分布
- GPU 型号、显存与训练耗时
- 是否保存最佳权重与测试集评估结果
