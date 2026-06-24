import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from detect_illegal_parking import (  # noqa: E402
    BUILDING_CLASS,
    PAVEMENT_CLASS,
    ROAD_CLASS,
    VEHICLE_CLASS,
    compute_relation_features,
)


class ParkingRelationScoringTest(unittest.TestCase):
    def test_vehicle_surrounded_by_road_has_low_risk(self):
        mask = np.full((80, 80), ROAD_CLASS, dtype=np.uint8)
        vehicle_mask = np.zeros_like(mask, dtype=bool)
        vehicle_mask[30:45, 35:50] = True
        mask[vehicle_mask] = VEHICLE_CLASS

        features = compute_relation_features(mask, vehicle_mask, x=35, y=30, w=15, h=15, margin=12)

        self.assertGreater(features.road_context_ratio, 0.80)
        self.assertLess(features.pavement_context_ratio, 0.05)
        self.assertLess(features.relation_score, 0.35)
        self.assertFalse(features.is_illegal)

    def test_vehicle_on_pavement_has_high_risk(self):
        mask = np.full((80, 80), ROAD_CLASS, dtype=np.uint8)
        mask[:, :50] = PAVEMENT_CLASS
        vehicle_mask = np.zeros_like(mask, dtype=bool)
        vehicle_mask[30:45, 20:35] = True
        mask[vehicle_mask] = VEHICLE_CLASS

        features = compute_relation_features(mask, vehicle_mask, x=20, y=30, w=15, h=15, margin=12)

        self.assertGreater(features.pavement_context_ratio, 0.60)
        self.assertGreater(features.relation_score, 0.55)
        self.assertTrue(features.is_illegal)

    def test_vehicle_next_to_building_and_pavement_is_suspicious(self):
        mask = np.full((80, 80), ROAD_CLASS, dtype=np.uint8)
        mask[:, :30] = BUILDING_CLASS
        mask[:, 30:48] = PAVEMENT_CLASS
        vehicle_mask = np.zeros_like(mask, dtype=bool)
        vehicle_mask[30:45, 38:53] = True
        mask[vehicle_mask] = VEHICLE_CLASS

        features = compute_relation_features(mask, vehicle_mask, x=38, y=30, w=15, h=15, margin=12)

        self.assertGreater(features.pavement_context_ratio, 0.25)
        self.assertGreater(features.static_obstacle_context_ratio, 0.10)
        self.assertGreater(features.relation_score, 0.40)


if __name__ == "__main__":
    unittest.main()
