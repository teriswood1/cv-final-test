import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CAMVID_CLASS_NAMES = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
    "Unlabelled",
]


def literal_constant(path: str, name: str):
    tree = ast.parse((PROJECT_ROOT / path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def imported_modules(path: str) -> set[str]:
    tree = ast.parse((PROJECT_ROOT / path).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def source_text(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


class ClassMappingTest(unittest.TestCase):
    def test_segformer_uses_camvid_12_class_mapping(self):
        self.assertEqual(literal_constant("train_segformer.py", "CLASS_NAMES"), CAMVID_CLASS_NAMES)
        self.assertEqual(literal_constant("train_segformer.py", "NUM_CLASSES"), 12)

    def test_inference_and_label_audit_use_same_mapping(self):
        self.assertEqual(literal_constant("infer_segformer.py", "NUM_CLASSES"), 12)
        self.assertEqual(literal_constant("check_train_labels.py", "CLASS_NAMES"), CAMVID_CLASS_NAMES)
        self.assertEqual(literal_constant("visualize_label_issues.py", "NUM_CLASSES"), 12)

    def test_illegal_parking_uses_vehicle_and_road_class_ids(self):
        self.assertEqual(literal_constant("detect_illegal_parking.py", "ROAD_CLASS"), 3)
        self.assertEqual(literal_constant("detect_illegal_parking.py", "PAVEMENT_CLASS"), 4)
        self.assertEqual(literal_constant("detect_illegal_parking.py", "VEHICLE_CLASS"), 8)
        self.assertEqual(literal_constant("detect_illegal_parking.py", "PERSON_CLASS"), 9)

    def test_training_and_inference_use_fresh_camvid12_outputs(self):
        self.assertEqual(literal_constant("train_segformer.py", "OUTPUT_DIR"), "outputs/segformer_camvid12_vsra")
        self.assertEqual(literal_constant("infer_segformer.py", "CHECKPOINT_PATH"), "outputs/segformer_camvid12_vsra/best_model.pth")
        self.assertEqual(literal_constant("infer_segformer.py", "OUTPUT_DIR"), "outputs/predictions_camvid12_vsra_train")

    def test_training_and_inference_enable_relation_attention(self):
        self.assertTrue(literal_constant("train_segformer.py", "ENABLE_RELATION_ATTENTION"))
        self.assertTrue(literal_constant("infer_segformer.py", "ENABLE_RELATION_ATTENTION"))
        self.assertEqual(literal_constant("train_segformer.py", "RELATION_CHANNELS"), 64)
        self.assertEqual(literal_constant("infer_segformer.py", "RELATION_POOL_SIZE"), 16)
        self.assertIn("add_vehicle_scene_relation_attention", source_text("train_segformer.py"))
        self.assertIn("add_vehicle_scene_relation_attention", source_text("infer_segformer.py"))

    def test_training_script_avoids_unavailable_image_dependencies(self):
        modules = imported_modules("train_segformer.py")
        self.assertNotIn("albumentations", modules)
        self.assertNotIn("cv2", modules)

    def test_training_script_resumes_from_last_checkpoint(self):
        source = source_text("train_segformer.py")
        self.assertIn("last_model_path", source)
        self.assertIn("model.load_state_dict", source)
        self.assertIn("optimizer.load_state_dict", source)
        self.assertIn("start_epoch", source)
        self.assertIn("append_log", source)

    def test_training_script_loads_pretrained_model_offline(self):
        source = source_text("train_segformer.py")
        self.assertIn("local_files_only=True", source)
        self.assertIn("SegformerConfig", source)
        self.assertIn("pretrained: bool", source)

    def test_inference_script_builds_model_offline(self):
        source = source_text("infer_segformer.py")
        self.assertIn("SegformerConfig", source)
        self.assertIn("local_files_only=True", source)


if __name__ == "__main__":
    unittest.main()
