from dataclasses import dataclass


@dataclass(frozen=True)
class LabelProtocol:
    name: str
    num_label_ids: int
    train_num_classes: int
    ignore_index: int | None = None


CAMVID_12_ID_PROTOCOL = LabelProtocol(
    name="camvid_12_ids_configurable_void",
    num_label_ids=12,
    train_num_classes=12,
    ignore_index=None,
)
