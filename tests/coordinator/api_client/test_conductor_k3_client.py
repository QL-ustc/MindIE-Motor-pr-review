# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""K3 namespace and restart reconciliation behavior for ConductorApiClient."""

from unittest.mock import Mock, patch

import pytest

from motor.common.resources.instance import Endpoint, Instance, PDRole
from motor.coordinator.api_client.conductor_api_client import ConductorApiClient
from motor.coordinator.api_client.conductor_evidence import ConductorSchemaError
from motor.config.coordinator import KvConductorConfig, SchedulerConfig


def _instance(instance_id=1, model="qwen", ranks=(0,)):
    endpoints = {"pod-0": {rank: Endpoint(id=rank, ip="10.0.0.1", business_port=str(8000 + rank)) for rank in ranks}}
    return Instance(
        id=instance_id,
        role=PDRole.ROLE_P,
        model_name=model,
        job_name="test-job",
        endpoints=endpoints,
    )


def _config(store_backend="Mooncake"):
    reg = KvConductorConfig(
        store_backend=store_backend,
        pool_endpoint="tcp://pool:5557",
        npu_endpoint="tcp://*:50090",
        model_path="stale-config-model",
        block_size=128,
        conductor_service="kv-conductor",
        http_server_port=13333,
    )
    config = Mock()
    config.scheduler_config = SchedulerConfig(kv_conductor_config=reg)
    return config


def _worker(instance_id, model, ranks, *, provenance=True):
    return {
        "instance_id": instance_id,
        "model_name": model,
        "tenant_id": "default",
        "block_size": 128,
        "endpoints": {
            str(rank): {
                "medium_endpoints": {"npu": f"tcp://10.0.0.1:{50090 + rank}"},
                "engine_type": "vllm",
                "dp_rank": rank,
                "replay_endpoint": None,
                "exact_dp_npu_provenance": provenance,
            }
            for rank in ranks
        },
    }


def test_pool_and_dp_registration_share_runtime_model_namespace():
    client = Mock()
    client.post.return_value = {"status": "ok"}
    ConductorApiClient._pool_registered = False

    with (
        patch.object(ConductorApiClient, "coordinator_config", _config()),
        patch("motor.coordinator.api_client.conductor_api_client.SafeHTTPSClient") as http,
    ):
        http.return_value.__enter__.return_value = client
        ConductorApiClient.register_kv_instance([_instance(model="qwen")])

    payloads = [call.args[1] for call in client.post.call_args_list]
    assert len(payloads) == 2
    assert {payload["modelname"] for payload in payloads} == {"qwen"}


def test_mixed_runtime_models_fail_closed_before_registration():
    with (
        patch.object(ConductorApiClient, "coordinator_config", _config()),
        patch("motor.coordinator.api_client.conductor_api_client.SafeHTTPSClient") as http,
    ):
        ConductorApiClient.register_kv_instance([_instance(1, "qwen"), _instance(2, "llama")])

    http.assert_not_called()


def test_missing_runtime_model_fails_closed_before_registration():
    instance = _instance(model="qwen")
    instance.model_name = ""

    with (
        patch.object(ConductorApiClient, "coordinator_config", _config()),
        patch("motor.coordinator.api_client.conductor_api_client.SafeHTTPSClient") as http,
    ):
        ConductorApiClient.register_kv_instance([instance])

    http.assert_not_called()


def test_workers_endpoint_is_strict_and_never_falls_back_to_legacy_services():
    client = Mock()
    client.get.return_value = {"services": [{"InstanceID": "legacy", "DPRank": 0}]}

    with (
        patch.object(ConductorApiClient, "coordinator_config", _config()),
        patch("motor.coordinator.api_client.conductor_api_client.SafeHTTPSClient") as http,
    ):
        http.return_value.__enter__.return_value = client
        with pytest.raises(ConductorSchemaError):
            ConductorApiClient.get_registered_services()

    client.get.assert_called_once_with("/workers")


def test_conductor_restart_rebuilds_pool_and_every_missing_dp_cold():
    instance = _instance(model="qwen", ranks=(0, 1))
    ConductorApiClient._pool_registered = True

    with (
        patch.object(ConductorApiClient, "coordinator_config", _config()),
        patch.object(ConductorApiClient, "get_registered_services", return_value=[]),
        patch.object(ConductorApiClient, "_register_pool") as register_pool,
        patch.object(ConductorApiClient, "register_post") as register_dp,
    ):
        ConductorApiClient.re_register_kv_instances([instance])

    register_pool.assert_called_once()
    assert register_pool.call_args.args[2] == "qwen"
    assert {call.args[2] for call in register_dp.call_args_list} == {0, 1}


def test_namespace_conflict_aborts_reconciliation_without_mutation():
    instance = _instance(model="qwen")
    conflict = _worker("vllm-prefill-1", "other-model", (0,))

    with (
        patch.object(ConductorApiClient, "coordinator_config", _config("YuanRong")),
        patch.object(ConductorApiClient, "get_registered_services", return_value=[conflict]),
        patch.object(ConductorApiClient, "register_post") as register_dp,
    ):
        ConductorApiClient.re_register_kv_instances([instance])

    register_dp.assert_not_called()
