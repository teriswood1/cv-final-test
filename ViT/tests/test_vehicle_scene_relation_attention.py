import sys
import unittest
from pathlib import Path

import torch
from transformers import SegformerConfig, SegformerForSemanticSegmentation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class VehicleSceneRelationAttentionTest(unittest.TestCase):
    def test_relation_attention_preserves_feature_shape_and_gradients(self):
        from models.vehicle_scene_relation_attention import VehicleSceneRelationAttention

        module = VehicleSceneRelationAttention(in_channels=16, relation_channels=8, pool_size=4)
        features = torch.randn(2, 16, 12, 10, requires_grad=True)

        enhanced = module(features)
        loss = enhanced.mean()
        loss.backward()

        self.assertEqual(enhanced.shape, features.shape)
        self.assertIsNotNone(features.grad)
        self.assertEqual(features.grad.shape, features.shape)

    def test_wrap_segformer_adds_relation_module_before_classifier(self):
        from models.vehicle_scene_relation_attention import (
            RelationEnhancedClassifier,
            add_vehicle_scene_relation_attention,
        )

        config = SegformerConfig(
            num_labels=12,
            depths=[1, 1, 1, 1],
            hidden_sizes=[8, 16, 32, 64],
            decoder_hidden_size=16,
            num_attention_heads=[1, 2, 4, 8],
            sr_ratios=[8, 4, 2, 1],
        )
        model = SegformerForSemanticSegmentation(config)

        wrapped = add_vehicle_scene_relation_attention(
            model,
            relation_channels=8,
            pool_size=4,
        )

        self.assertIs(wrapped, model)
        self.assertIsInstance(model.decode_head.classifier, RelationEnhancedClassifier)

        outputs = model(pixel_values=torch.randn(1, 3, 32, 32))
        self.assertEqual(outputs.logits.shape[1], 12)


if __name__ == "__main__":
    unittest.main()
