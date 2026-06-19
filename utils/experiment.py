def experiment_name_with_epochs(base_name, num_epochs):
    """Return a stable experiment prefix that preserves epoch-specific outputs."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    return f"{base_name}_e{num_epochs}"
