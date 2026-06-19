from utils.experiment import experiment_name_with_epochs


def test_experiment_name_with_epochs_appends_epoch_suffix():
    assert experiment_name_with_epochs("unet_aspp_fixed_aug", 50) == (
        "unet_aspp_fixed_aug_e50"
    )


def test_experiment_name_with_epochs_rejects_non_positive_epochs():
    try:
        experiment_name_with_epochs("unet_aspp_fixed_aug", 0)
    except ValueError as exc:
        assert "num_epochs must be positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
