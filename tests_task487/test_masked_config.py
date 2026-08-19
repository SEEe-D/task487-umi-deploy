from openpi.training import config as config_lib


def test_masked_task487_config_matches_cloud_training_inputs_and_deployment_contract():
    config = config_lib.get_config("pi05_umi_task487_masked_12_5")

    assert config.model.pi05 is True
    assert config.model.action_horizon == 20
    assert config.model.image_geometric_augmentation is False
    assert config.data.head_feature == "observation.images.head_fixed"
    assert config.data.use_head_camera is True
    assert config.data.use_head_mask is True
    assert config.policy_metadata["runtime"] == "pi05_umi_task487_masked_12_5_v1"
    assert config.policy_metadata["mask_enabled"] is True
    assert config.policy_metadata["rtc_prefix_steps"] == 5
