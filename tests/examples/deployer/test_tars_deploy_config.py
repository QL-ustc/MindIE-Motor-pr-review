# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Deployment-envelope tests for public TARS examples."""

import copy
import json
from pathlib import Path

import deploy as deploy_module
import pytest
import yaml
from lib.config_validator import validate_tars_deploy_config
from lib.generator import k8s_utils

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATHS = (
    REPO_ROOT / "examples" / "features" / "tars_p4d4" / "user_config.json",
    REPO_ROOT / "examples" / "features" / "tars_kv_pool_dp2" / "user_config.json",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", EXAMPLE_PATHS)
def test_checked_in_tars_examples_meet_initial_support_envelope(path):
    validate_tars_deploy_config(_load(path))


def test_tars_rejects_master_standby():
    config = _load(EXAMPLE_PATHS[0])
    config["motor_coordinator_config"]["standby_config"]["enable_master_standby"] = True

    with pytest.raises(ValueError, match="one active Coordinator"):
        validate_tars_deploy_config(config)


def test_tars_rejects_mismatched_engine_block_size():
    config = _load(EXAMPLE_PATHS[0])
    config["motor_engine_decode_config"]["engine_config"]["block-size"] = 64

    with pytest.raises(ValueError, match="one block size"):
        validate_tars_deploy_config(config)


def test_tars_rejects_multiple_model_identities():
    config = copy.deepcopy(_load(EXAMPLE_PATHS[0]))
    config["motor_engine_decode_config"]["engine_config"]["served_model_name"] = "other-model"

    with pytest.raises(ValueError, match="exactly one model identity"):
        validate_tars_deploy_config(config)


@pytest.mark.parametrize("legacy_type", [None, "tars"])
def test_non_tars_config_is_unchanged_by_tars_validator(legacy_type):
    config = _load(EXAMPLE_PATHS[0])
    scheduler = config["motor_coordinator_config"]["scheduler_config"]
    scheduler["prefill_scheduler_type"] = "load_balance"
    scheduler["decode_scheduler_type"] = "round_robin"
    if legacy_type is not None:
        scheduler["scheduler_type"] = legacy_type
    config.pop("kv_conductor_config")
    before = copy.deepcopy(config)

    validate_tars_deploy_config(config)

    assert config == before


@pytest.mark.parametrize(
    "scheduler_config",
    [
        {"prefill_scheduler_type": "tars"},
        {"decode_scheduler_type": "tars"},
        {"scheduler_type": "tars", "prefill_scheduler_type": "round_robin"},
        {"scheduler_type": "tars", "decode_scheduler_type": "load_balance"},
    ],
)
def test_tars_deployer_rejects_unpaired_role_configuration(scheduler_config):
    config = _load(EXAMPLE_PATHS[0])
    config["motor_coordinator_config"]["scheduler_config"] = scheduler_config

    with pytest.raises(ValueError, match="prefill_scheduler_type and decode_scheduler_type"):
        validate_tars_deploy_config(config)


def test_legacy_tars_default_still_enforces_the_deployment_envelope():
    config = _load(EXAMPLE_PATHS[0])
    config["motor_coordinator_config"]["scheduler_config"] = {"scheduler_type": "tars"}
    config["motor_coordinator_config"]["standby_config"]["enable_master_standby"] = True

    with pytest.raises(ValueError, match="one active Coordinator"):
        validate_tars_deploy_config(config)


def _deploy_paths(output_dir: Path) -> dict[str, str]:
    templates = REPO_ROOT / "examples" / "deployer" / "yaml_template"
    return {
        "controller_input_yaml": str(templates / "controller_template.yaml"),
        "controller_output_yaml": str(output_dir / "mindie_motor_controller.yaml"),
        "coordinator_input_yaml": str(templates / "coordinator_template.yaml"),
        "coordinator_output_yaml": str(output_dir / "mindie_motor_coordinator.yaml"),
        "engine_input_yaml": str(templates / "engine_template.yaml"),
        "engine_output_yaml": str(output_dir / "mindie-server"),
        "kv_store_input_yaml": str(templates / "kv_cache_store_template.yaml"),
        "kv_store_output_yaml": str(output_dir / "mindie_motor_kv_store.yaml"),
        "kv_conductor_input_yaml": str(templates / "kv_conductor_template.yaml"),
        "kv_conductor_output_yaml": str(output_dir / "mindie_motor_kv_conductor.yaml"),
        "infer_service_input_yaml": str(templates / "infer_service_template.yaml"),
        "infer_service_output_yaml": str(output_dir / "infer_service.yaml"),
        "single_container_input_yaml": str(templates / "single_container_template.yaml"),
        "single_container_output_yaml": str(output_dir / "mindie_motor_single_container.yaml"),
        "mf_store_input_yaml": str(templates / "mf_store_template.yaml"),
        "mf_store_output_yaml": str(output_dir / "mindie_motor_mf_store.yaml"),
        "storage_pvc_input_yaml": str(templates / "storage_pvc_template.yaml"),
        "storage_pvc_output_yaml": str(output_dir / "mindie_motor_storage_pvc.yaml"),
    }


@pytest.mark.parametrize(
    ("path", "prefill_count", "decode_count"),
    [
        (EXAMPLE_PATHS[0], 4, 4),
        (EXAMPLE_PATHS[1], 2, 2),
    ],
)
def test_tars_examples_generate_complete_dry_run_manifests(
    path,
    prefill_count,
    decode_count,
    tmp_path,
    monkeypatch,
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(deploy_module, "get_deploy_paths", lambda: _deploy_paths(output_dir))
    monkeypatch.setattr(deploy_module, "resolve_nodeports_for_yaml_files", lambda *_args, **_kwargs: None)
    k8s_utils.g_generate_yaml_list = []

    deploy_module.deploy_services(_load(path), env_config_path=None, dry_run=True)

    generated = [Path(item) for item in k8s_utils.g_generate_yaml_list]
    assert sum(item.name.startswith("mindie-server_p") for item in generated) == prefill_count
    assert sum(item.name.startswith("mindie-server_d") for item in generated) == decode_count
    assert any(item.name == "mindie_motor_kv_conductor.yaml" for item in generated)
    for manifest in generated:
        documents = list(yaml.safe_load_all(manifest.read_text(encoding="utf-8")))
        assert documents
        assert all(isinstance(document, dict) for document in documents)

    coordinator = list(yaml.safe_load_all((output_dir / "mindie_motor_coordinator.yaml").read_text(encoding="utf-8")))
    conductor = list(yaml.safe_load_all((output_dir / "mindie_motor_kv_conductor.yaml").read_text(encoding="utf-8")))
    coordinator_deployment = next(item for item in coordinator if item.get("kind") == "Deployment")
    conductor_deployment = next(item for item in conductor if item.get("kind") == "Deployment")
    assert coordinator_deployment["spec"]["replicas"] == 1
    assert conductor_deployment["spec"]["replicas"] == 1
