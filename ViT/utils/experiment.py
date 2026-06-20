from pathlib import Path


def experiment_name_with_epochs(base_name: str, num_epochs: int) -> str:
    return f"{base_name}_e{num_epochs}"
