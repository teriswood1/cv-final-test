from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class VehicleSceneRelationAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        relation_channels: int = 64,
        pool_size: int = 16,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if relation_channels <= 0:
            raise ValueError("relation_channels must be positive")
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")

        self.pool_size = pool_size
        self.query = nn.Conv2d(in_channels, relation_channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(in_channels, relation_channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(in_channels, relation_channels, kernel_size=1, bias=False)
        self.vehicle_gate = nn.Sequential(
            nn.Conv2d(in_channels, relation_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(relation_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            nn.Conv2d(relation_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must have shape [B, C, H, W]")

        batch_size, _, height, width = features.shape
        query = self.query(features).flatten(2).transpose(1, 2)

        scene_features = F.adaptive_avg_pool2d(features, output_size=(self.pool_size, self.pool_size))
        key = self.key(scene_features).flatten(2)
        value = self.value(scene_features).flatten(2).transpose(1, 2)

        scale = math.sqrt(key.shape[1])
        attention = torch.softmax(torch.bmm(query, key) / scale, dim=-1)
        relation = torch.bmm(attention, value)
        relation = relation.transpose(1, 2).reshape(batch_size, -1, height, width)

        gated_relation = relation * self.vehicle_gate(features)
        enhanced = self.output(gated_relation)
        return features + self.gamma * enhanced


class RelationEnhancedClassifier(nn.Module):
    def __init__(
        self,
        classifier: nn.Conv2d,
        relation_channels: int = 64,
        pool_size: int = 16,
    ) -> None:
        super().__init__()
        if not isinstance(classifier, nn.Conv2d):
            raise TypeError("classifier must be an nn.Conv2d layer")
        self.relation_attention = VehicleSceneRelationAttention(
            in_channels=classifier.in_channels,
            relation_channels=relation_channels,
            pool_size=pool_size,
        )
        self.classifier = classifier

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.relation_attention(features))


def add_vehicle_scene_relation_attention(
    model: nn.Module,
    relation_channels: int = 64,
    pool_size: int = 16,
) -> nn.Module:
    classifier = model.decode_head.classifier
    if isinstance(classifier, RelationEnhancedClassifier):
        return model
    model.decode_head.classifier = RelationEnhancedClassifier(
        classifier=classifier,
        relation_channels=relation_channels,
        pool_size=pool_size,
    )
    return model
