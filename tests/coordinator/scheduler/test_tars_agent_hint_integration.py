# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""TARS and Agent Hint integration; only transport, evidence and native I/O are doubles."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, create_autospec

import pytest

from motor.common.resources.endpoint import Endpoint, EndpointStatus, Workload, WorkloadAction
from motor.common.resources.http_msg_spec import EventType
from motor.common.resources.instance import InsStatus, Instance, ParallelConfig, PDRole
from motor.config.coordinator import CoordinatorConfig, KvcLifecycleConfig, SchedulerType
from motor.coordinator.domain.agent_hint import parse_agent_hint
from motor.coordinator.domain.agent_session_status import AgentSessionStateMachine
from motor.coordinator.domain.cache_declaration import CacheDeclarationTable
from motor.coordinator.domain.instance_manager import InstanceManager
from motor.coordinator.domain.scheduling import UpdateWorkloadParams
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.policy.tars.models import CacheCreditCapability, TarsCacheEvidence
from motor.coordinator.scheduler.policy.tars.original.models import TarsTarget
from motor.coordinator.scheduler.runtime import scheduler_client as client_module
from motor.coordinator.scheduler.runtime.agent_hint_lifecycle import CentralAgentLifecycle
from motor.coordinator.scheduler.runtime.scheduler_client import AsyncSchedulerClient, SchedulerClientConfig
from motor.coordinator.scheduler.runtime.scheduler_server import _SchedulerRequestDispatcher
from motor.coordinator.scheduler.runtime.session_affinity import SessionAffinityTable
from motor.coordinator.scheduler.runtime.workload_shm.layout import FLAG_VALID, SCHEMA_VERSION
from motor.coordinator.scheduler.runtime.workload_shm.native import (
    STATUS_CHANGED,
    STATUS_OK,
    STATUS_SLOT_INVALID,
    WorkloadShm,
)
from motor.coordinator.scheduler.runtime.zmq_protocol import (
    SchedulerRequestType,
    SchedulerResponse,
    SchedulerResponseType,
)
from motor.coordinator.scheduler.scheduler import Scheduler


SESSION = "agent-session"
AGENT_EPOCH = 4


def _instance(instance_id, role, endpoint_id):
    instance = Instance(
        id=instance_id,
        job_name=f"agent-{instance_id}",
        model_name="test-model",
        engine_type="sglang",
        role=role,
        status=InsStatus.ACTIVE,
        parallel_config=ParallelConfig(dp_size=1),
    )
    instance.add_endpoints(
        f"pod-{instance_id}",
        {
            endpoint_id: Endpoint(
                id=endpoint_id,
                ip=f"10.0.0.{instance_id}",
                business_port=str(8000 + instance_id),
                status=EndpointStatus.NORMAL,
                workload=Workload(),
            ),
        },
    )
    return instance


def _native(instances):
    roles = {PDRole.ROLE_P: 0, PDRole.ROLE_D: 1, PDRole.ROLE_U: 2}
    entries = {
        (instance.id, endpoint.id): {
            "instance_id": instance.id,
            "endpoint_id": endpoint.id,
            "role": roles[PDRole(instance.role)],
            "flags": FLAG_VALID,
            "generation": 7,
            "active_tokens": 0.0,
            "request_count": 0,
            "total_requests": 0,
        }
        for instance in instances
        for endpoint in instance.get_all_endpoints()
    }
    native = create_autospec(WorkloadShm, instance=True, spec_set=True)
    native.read_header.return_value = {"schema_version": SCHEMA_VERSION, "sequence": 2, "entry_count": len(entries)}
    native.load_entries.side_effect = lambda _count: [dict(entry) for entry in entries.values()]

    def add(instance_id, endpoint_id, generation, expected, delta, slot=None):
        assert slot is None
        entry = entries[(instance_id, endpoint_id)]
        if generation != entry["generation"]:
            return STATUS_SLOT_INVALID, entry["active_tokens"]
        if expected != entry["active_tokens"]:
            return STATUS_CHANGED, entry["active_tokens"]
        entry["active_tokens"] += delta
        entry["request_count"] += 1
        entry["total_requests"] += 1
        return STATUS_OK, entry["active_tokens"]

    def subtract(instance_id, endpoint_id, generation, delta, slot=None):
        assert slot is None
        entry = entries[(instance_id, endpoint_id)]
        if generation != entry["generation"]:
            return STATUS_SLOT_INVALID, entry["active_tokens"]
        entry["active_tokens"] = max(0.0, entry["active_tokens"] - delta)
        entry["request_count"] = max(0, entry["request_count"] - 1)
        return STATUS_OK, entry["active_tokens"]

    native.cas_add.side_effect = add
    native.cas_sub_floor0.side_effect = subtract
    return native, entries


def _request(*, explicit_ttl=True):
    cache_control = {"type": "ephemeral", **({"ttl": 120} if explicit_ttl else {})}
    body = {"messages": [{"role": "user", "content": "test"}], "agent_hint": {"cache_control": cache_control}}
    hint = parse_agent_hint(body, header_hint={"session": {"session_id": SESSION, "epoch": AGENT_EPOCH}})
    return RequestInfo(
        req_id="agent-request",
        req_data={"max_tokens": 8},
        req_len=4,
        token_ids=[1, 2, 3, 4],
        api="v1/chat/completions",
        agent_hint_info=hint,
    )


async def _setup(monkeypatch, *, confirmation="success"):
    config = CoordinatorConfig()
    config.scheduler_config.prefill_scheduler_type = SchedulerType.TARS
    config.scheduler_config.decode_scheduler_type = SchedulerType.TARS
    manager = InstanceManager(config)
    instances = [
        _instance(1, PDRole.ROLE_P, 7),
        _instance(2, PDRole.ROLE_P, 8),
        _instance(3, PDRole.ROLE_D, 9),
        _instance(4, PDRole.ROLE_U, 11),
    ]
    await manager.refresh_instances(EventType.ADD, instances)
    native, entries = _native(instances)
    dispatcher = _SchedulerRequestDispatcher(
        manager,
        Scheduler(manager, config),
        config,
        workload_writer=SimpleNamespace(native=native),
    )

    def evidence(eligible, data):
        return tuple(
            TarsCacheEvidence(
                target=TarsTarget(str(instance.id), endpoint.id),
                input_tokens=len(data["token_ids"]),
                capability=CacheCreditCapability.DP_NPU_ONLY,
                observed_exact_dp_npu_tokens=0,
                best_effort_cache_credit_tokens=0,
                estimated_compute_tokens=len(data["token_ids"]),
                diagnostic_matched_tokens=0,
                diagnostic_cpu_blocks=0,
                diagnostic_disk_blocks=0,
            )
            for instance in eligible
            for endpoint in instance.get_all_endpoints()
        )

    monkeypatch.setattr(dispatcher._tars_activation, "_build_evidence", evidence)
    tls = object()
    client = AsyncSchedulerClient(
        SchedulerClientConfig(
            scheduler_type="tars",
            kvc_lifecycle=KvcLifecycleConfig(enabled=True, manage_timeout_s=0.5),
            tls_config=tls,
        )
    )
    client._session_affinity = SessionAffinityTable()
    client._cache_declarations = CacheDeclarationTable()
    await client._cache.apply_add(instances)

    async def transport(request):
        if request.request_type == SchedulerRequestType.TARS_CONFIRM:
            if confirmation == "cancel":
                raise asyncio.CancelledError()
            if confirmation == "reject":
                return SchedulerResponse(SchedulerResponseType.ERROR, request.request_id, error="confirmation rejected")
        return await dispatcher.dispatch(request)

    monkeypatch.setattr(client._transport, "send_request", transport)
    return client, dispatcher, native, entries, tls


async def _allocate(client, mode, request):
    if mode == "joint":
        return await client.tars_allocate_bootstrap_joint(request, attempt_seq=1)
    role = PDRole.ROLE_U if mode == "unified" else PDRole.ROLE_P
    allocation = await client.select_and_allocate(
        role, request, attempt_seq=1, coordination_mode=mode, required_engine_type="sglang"
    )
    return None if allocation is None else (allocation,)


@pytest.mark.parametrize("mode", ["handoff", "unified", "joint"])
@pytest.mark.parametrize("route", ["0", "1"])
async def test_confirmed_tars_records_actual_location_epoch_and_explicit_retention(monkeypatch, mode, route):
    monkeypatch.setenv("MOTOR_SESSION_AFFINITY_ROUTE", route)
    client, _dispatcher, _native_api, entries, _tls = await _setup(monkeypatch)
    role = PDRole.ROLE_U if mode == "unified" else PDRole.ROLE_P
    client._session_affinity.record(SESSION, 999, role.value, epoch=AGENT_EPOCH, endpoint_id=998)

    def unexpected_route(*_args, **_kwargs):
        pytest.fail("A session pin must not replace the TARS target")

    monkeypatch.setattr(client_module, "resolve_session_pin", unexpected_route)
    resources = await _allocate(client, mode, _request())

    assert resources is not None
    assert resources[0][0].id in ({4} if mode == "unified" else {1, 2})
    roles = set()
    for instance, endpoint, _workload in resources:
        role_value = PDRole(instance.role).value
        roles.add(role_value)
        pin = client._session_affinity.get(SESSION, role_value)
        assert (pin.instance_id, pin.endpoint_id, pin.epoch) == (instance.id, endpoint.id, AGENT_EPOCH)
    declaration = client._cache_declarations.get(SESSION)
    assert declaration is not None
    assert declaration.ttl_s == 120
    assert set(declaration.roles) == roles
    assert sum(entry["request_count"] for entry in entries.values()) == len(resources)
    assert await client.tars_abort("agent-request", 1)


@pytest.mark.parametrize("mode", ["handoff", "unified", "joint"])
@pytest.mark.parametrize("confirmation", ["reject", "cancel"])
async def test_failed_confirmation_creates_no_location_or_retention(monkeypatch, mode, confirmation):
    client, _dispatcher, _native_api, entries, _tls = await _setup(monkeypatch, confirmation=confirmation)
    if confirmation == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await _allocate(client, mode, _request())
    elif mode == "joint":
        with pytest.raises(RuntimeError, match="confirmation failed"):
            await _allocate(client, mode, _request())
    else:
        assert await _allocate(client, mode, _request()) is None
    assert len(client._session_affinity) == 0
    assert len(client._cache_declarations) == 0
    assert all(entry["request_count"] == 0 and entry["active_tokens"] == 0 for entry in entries.values())


def _http_boundary(monkeypatch):
    response = Mock()
    response.json.return_value = {"agent_hint_response": {"edit_results": [{"op_status": True}]}}
    http = SimpleNamespace(post=AsyncMock(return_value=response))
    pool = SimpleNamespace(get_client=AsyncMock(return_value=http))
    monkeypatch.setattr(client_module, "HTTPClientPool", lambda: pool)
    return pool, http


@pytest.mark.parametrize("mode", ["handoff", "unified", "joint"])
async def test_tars_locations_drive_real_kvc_sender_at_http_boundary(monkeypatch, mode):
    client, _dispatcher, _native_api, _entries, tls = await _setup(monkeypatch)
    resources = await _allocate(client, mode, _request())
    pool, http = _http_boundary(monkeypatch)

    results = await client.send_session_manage_request(SESSION, "evict")

    assert results == [{"op_status": True}] * len(resources)
    assert {(call.args[0], call.args[1]) for call in pool.get_client.await_args_list} == {
        (endpoint.ip, endpoint.business_port) for _instance_obj, endpoint, _workload in resources
    }
    assert all(call.kwargs["tls_config"] is tls for call in pool.get_client.await_args_list)
    assert http.post.await_count == len(resources)
    for call in http.post.await_args_list:
        assert call.args == ("/v1/chat/completions",)
        body = call.kwargs["json"]
        assert body["model"] == "test-model"
        assert body["agent_hint"]["session_id"] == SESSION
        assert body["agent_hint"]["context_management"] == {
            "manage_request": True,
            "edits": [{"type": "evict", "target": "session"}],
        }
    assert await client.tars_abort("agent-request", 1)


@pytest.mark.parametrize("mode", ["handoff", "joint"])
async def test_finalize_manages_kv_before_dropping_pins_and_preserves_workload_receipts(monkeypatch, mode):
    client, _dispatcher, native, entries, _tls = await _setup(monkeypatch)
    resources = await _allocate(client, mode, _request())
    _pool, http = _http_boundary(monkeypatch)
    machine = AgentSessionStateMachine()
    machine.apply(SESSION, "running", epoch=AGENT_EPOCH)
    client.agent_status_state_listener = machine.apply
    receipts = dict(client._tars_workload_receipts)
    central = CentralAgentLifecycle(kvc_config=KvcLifecycleConfig(enabled=True))
    central.set_sender(client.send_session_manage_request)

    async def finalize_release(session_id, epoch):
        assert http.post.await_count == len(resources)
        assert client._session_affinity.get(session_id, PDRole.ROLE_P.value) is not None
        await client._on_agent_status_event(session_id, "finalize", epoch)

    try:
        result = await central.apply(SESSION, "finalize", epoch=AGENT_EPOCH, finalize_release=finalize_release)
        assert result["applied"] is True
        assert len(client._session_affinity) == 0
        assert len(client._cache_declarations) == 0
        assert machine.get(SESSION).epoch == AGENT_EPOCH
        assert machine.get(SESSION).state == "finalize"
        assert client._tars_workload_receipts == receipts
        assert client.tars_has_attempt("agent-request", 1)
        assert sum(entry["request_count"] for entry in entries.values()) == len(resources)
        assert await client.tars_abort("agent-request", 1)
        for instance, endpoint, workload in resources:
            entry = entries[(instance.id, endpoint.id)]
            assert native.cas_add(instance.id, endpoint.id, 7, 0.0, 11.0)[0] == STATUS_OK
            params = UpdateWorkloadParams(
                instance_id=instance.id,
                endpoint_id=endpoint.id,
                role=PDRole(instance.role),
                req_id="agent-request",
                attempt_seq=1,
                workload_action=WorkloadAction.RELEASE_TOKENS,
                workload_change=Workload(active_tokens=-workload.active_tokens),
            )
            assert await client.update_workload(params)
            assert await client.update_workload(params)
            assert entry["active_tokens"] == 11.0
            assert entry["request_count"] == 1
    finally:
        await central.aclose()


async def test_implicit_ttl_records_location_without_declaring_retention(monkeypatch):
    client, _dispatcher, _native_api, _entries, _tls = await _setup(monkeypatch)
    assert await _allocate(client, "handoff", _request(explicit_ttl=False)) is not None
    assert client._session_affinity.get(SESSION, PDRole.ROLE_P.value) is not None
    assert len(client._cache_declarations) == 0
    assert await client.tars_abort("agent-request", 1)
