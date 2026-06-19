import torch
import numpy as np


class SegmentationMetric:
    """
    语义分割评价指标计算类。

    支持：
    1. Pixel Accuracy 像素准确率
    2. 每一类 IoU
    3. mIoU
    4. 混淆矩阵 confusion matrix
    """

    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """
        清空混淆矩阵。
        每次重新评估一个模型前都要 reset。
        """
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64
        )

    def _fast_hist(self, label_true, label_pred):
        """
        计算一张图的混淆矩阵。

        label_true: 真实标签，shape [H, W]
        label_pred: 预测标签，shape [H, W]

        混淆矩阵含义：
        行：真实类别
        列：预测类别
        """
        mask = (
            (label_true >= 0)
            & (label_true < self.num_classes)
        )
        if self.ignore_index is not None:
            mask &= label_true != self.ignore_index

        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int)
            + label_pred[mask].astype(int),
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        return hist

    def update(self, preds, labels):
        """
        更新混淆矩阵。

        preds 可以是两种形式：
        1. logits，shape [B, C, H, W]
        2. 已经 argmax 后的预测类别图，shape [B, H, W]

        labels:
        真实标签，shape [B, H, W]
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu()

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu()

        # 如果是 logits，则先 argmax 得到类别编号
        if preds.ndim == 4:
            preds = torch.argmax(preds, dim=1)

        preds = preds.numpy()
        labels = labels.numpy()

        for label_true, label_pred in zip(labels, preds):
            self.confusion_matrix += self._fast_hist(label_true, label_pred)

    def pixel_accuracy(self):
        """
        Pixel Accuracy = 所有预测正确的像素数 / 总像素数
        """
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()

        if total == 0:
            return 0.0

        return correct / total

    def class_iou(self):
        """
        计算每一类的 IoU。

        IoU = TP / (TP + FP + FN)

        对于混淆矩阵：
        TP = 对角线
        FP = 某一列总和 - TP
        FN = 某一行总和 - TP
        """
        hist = self.confusion_matrix

        intersection = np.diag(hist)
        union = (
            hist.sum(axis=1)
            + hist.sum(axis=0)
            - intersection
        )

        iou = intersection / np.maximum(union, 1)

        if self.ignore_index is None or not 0 <= self.ignore_index < self.num_classes:
            return iou

        return np.delete(iou, self.ignore_index)

    def mean_iou(self):
        """
        mIoU = 所有类别 IoU 的平均值。

        如果某个类别在真实标签和预测结果中都没有出现，
        这里不让它影响平均值。
        """
        iou = self.class_iou()

        valid = ~np.isnan(iou)

        if valid.sum() == 0:
            return 0.0

        return np.nanmean(iou[valid])

    def get_results(self):
        """
        返回所有评价结果。
        """
        pa = self.pixel_accuracy()
        iou = self.class_iou()
        miou = self.mean_iou()

        results = {
            "Pixel Accuracy": pa,
            "Mean IoU": miou,
            "Class IoU": iou,
            "Confusion Matrix": self.confusion_matrix,
        }

        return results


if __name__ == "__main__":
    """
    简单测试：
    假设有 3 个类别，测试指标计算是否能正常运行。
    """
    metric = SegmentationMetric(num_classes=3)

    labels = torch.tensor([
        [
            [0, 0, 1],
            [1, 2, 2],
            [0, 1, 2],
        ]
    ])

    preds = torch.tensor([
        [
            [0, 1, 1],
            [1, 2, 0],
            [0, 1, 2],
        ]
    ])

    metric.update(preds, labels)
    results = metric.get_results()

    print("Pixel Accuracy:", results["Pixel Accuracy"])
    print("Mean IoU:", results["Mean IoU"])
    print("Class IoU:", results["Class IoU"])
    print("Confusion Matrix:")
    print(results["Confusion Matrix"])
