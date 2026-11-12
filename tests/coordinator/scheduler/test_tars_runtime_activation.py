# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Integration tests for public TARS activation at the Scheduler boundary."""

import asyncio
import json
import threading
from itertools import count

import pytest

from motor.common.resources.endpoint import Endpoint, EndpointStatus, Workload, WorkloadAction
from motor.common.resources.http_msg_spec import EventType
from motor.common.resources.instance import InsStatus, Instance, ParallelConfig, PDRole
from motor.config.coordinator import CoordinatorConfig, SchedulerType
from motor.coordinator.domain.instance_manager import InstanceManager
from motor.coordinator.domain.scheduling import UpdateWorkloadParams
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.policy.tars.models import (
    CacheCreditCapability,
    TarsCacheEvidence,
)
from motor.coordinator.scheduler.policy.tars.original.models import TarsTarget
from motor.coordinator.scheduler.runtime import tars_activation as activation_module
from motor.coordinator.scheduler.runtime.scheduler_client import AsyncSchedulerClient, SchedulerClientConfig
from motor.coordinator.scheduler.runtime.scheduler_server import (
    _SchedulerRequestDispatcher,
)
from motor.coordinator.scheduler.runtime.workload_shm.layout import FLAG_VALID, SCHEMA_VERSION
from motor.coordinator.scheduler.runtime.workload_shm.native import (
    STATUS_BLOCKED,
    STATUS_CHANGED,
    STATUS_OK,
    STATUS_SLOT_INVALID,
)
from motor.coordinator.scheduler.runtime.zmq_protocol import (
    SchedulerRequest,
    SchedulerRequestType,
    SchedulerResponseType,
)
from motor.coordinator.scheduler.scheduler import Scheduler


class _Writer:
    """Native CAS boundary double; InstanceManager is intentionally not a workload ledger."""

    def __init__(self, manager) -> None:
        self.manager = manager
        self.native = self
        self.entries = {}
        self.blocked = set()
        self.fail_release = False
        self.writes: list[tuple[int, int, float]] = []

    def read_header(self):
        roles = {PDRole.ROLE_P: 0, PDRole.ROLE_D: 1, PDRole.ROLE_U: 2}
        for instance in self.manager.get_available_instances(None).values():
            for endpoint in instance.get_all_endpoints():
                self.entries.setdefault(
                    (instance.id, endpoint.id),
                    {
                        "instance_id": instance.id,
                        "endpoint_id": endpoint.id,
                        "role": roles[PDRole(instance.role)],
                        "flags": FLAG_VALID,
                        "generation": 1,
                        "active_tokens": 0.0,
                        "request_count": 0,
                        "total_requests": 0,
                    },
                )
        return {"schema_version": SCHEMA_VERSION, "sequence": 2, "entry_count": len(self.entries)}

    def load_entries(self, _count):
        return [dict(entry) for entry in self.entries.values()]

    def cas_add(self, instance_id, endpoint_id, generation, expected, delta):
        entry = self.entries.get((instance_id, endpoint_id))
        if entry is None or entry["generation"] != generation:
            return STATUS_SLOT_INVALID, 0.0
        if instance_id in self.blocked:
            return STATUS_BLOCKED, entry["active_tokens"]
        if entry["active_tokens"] != expected:
            return STATUS_CHANGED, entry["active_tokens"]
        entry["active_tokens"] += delta
        entry["request_count"] += 1
        entry["total_requests"] += 1
        self.writes.append((instance_id, endpoint_id, entry["active_tokens"]))
        return STATUS_OK, entry["active_tokens"]

    def cas_sub_floor0(self, instance_id, endpoint_id, generation, delta):
        if self.fail_release:
            return 7, 0.0
        entry = self.entries.get((instance_id, endpoint_id))
        if entry is None or entry["generation"] != generation:
            return STATUS_SLOT_INVALID, 0.0
        entry["active_tokens"] = max(0.0, entry["active_tokens"] - delta)
        entry["request_count"] = max(0, entry["request_count"] - 1)
        self.writes.append((instance_id, endpoint_id, entry["active_tokens"]))
        return STATUS_OK, entry["active_tokens"]


def _instance(instance_id: int, role: PDRole, dp_rank: int, engine_type: str = "sglang") -> Instance:
    instance = Instance(
        job_name=f"{role.value}-{instance_id}",
        model_name="test-model",
        engine_type=engine_type,
        id=instance_id,
        role=role,
        status=InsStatus.ACTIVE,
        parallel_config=ParallelConfig(dp_size=1),
    )
    instance.add_endpoints(
        f"pod-{instance_id}",
        {
            dp_rank: Endpoint(
                id=dp_rank,
                ip=f"10.0.0.{instance_id}",
                business_port=str(8000 + instance_id),
                status=EndpointStatus.NORMAL,
                workload=Workload(),
            )
        },
    )
    return instance


def _evidence(instances: list[Instance], input_tokens: int) -> tuple[TarsCacheEvidence, ...]:
    return tuple(
        TarsCacheEvidence(
            target=TarsTarget(str(instance.id), endpoint.id),
            input_tokens=input_tokens,
            capability=CacheCreditCapability.DP_NPU_ONLY,
            observed_exact_dp_npu_tokens=0,
            best_effort_cache_credit_tokens=0,
            estimated_compute_tokens=input_tokens,
            diagnostic_matched_tokens=0,
            diagnostic_cpu_blocks=0,
            diagnostic_disk_blocks=0,
        )
        for instance in instances
        for endpoint in instance.get_all_endpoints()
    )


async def _dispatcher(scheduler_type: SchedulerType, decode_type: SchedulerType | None = None):
    config = CoordinatorConfig()
    config.scheduler_config.prefill_scheduler_type = scheduler_type
    config.scheduler_config.decode_scheduler_type = decode_type or scheduler_type
    manager = InstanceManager(config)
    scheduler = Scheduler(manager, config)
    writer = _Writer(manager)
    dispatcher = _SchedulerRequestDispatcher(
        manager,
        scheduler,
        config,
        workload_writer=writer,
    )
    return dispatcher, manager, scheduler, writer


def _request(request_type: SchedulerRequestType, data: dict, suffix: str = "1") -> SchedulerRequest:
    return SchedulerRequest(request_type.value, f"transport-{suffix}", data)


def _admission_data(mode: str = "handoff") -> dict:
    return {
        "req_id": "request-1",
        "attempt_seq": 1,
        "session_id": "session-1",
        "token_ids": [1, 2, 3, 4],
        "max_output_tokens": 8,
        "decode_workload_tokens": 8,
        "coordination_mode": mode,
        "required_engine_type": "sglang",
    }


async def _add_topology(manager: InstanceManager) -> None:
    await manager.refresh_instances(
        EventType.ADD,
        [
            _instance(1, PDRole.ROLE_P, 0),
            _instance(2, PDRole.ROLE_P, 0),
            _instance(3, PDRole.ROLE_D, 0),
        ],
    )


async def _dispatch(dispatcher, request_type: SchedulerRequestType, data: dict, suffix: str = "1"):
    return await dispatcher.dispatch(_request(request_type, data, suffix))


@pytest.mark.parametrize("decode_type", [SchedulerType.LOAD_BALANCE, SchedulerType.ROUND_ROBIN])
async def test_tars_disabled_has_no_runtime_or_rpc_handler(decode_type):
    dispatcher, _manager, scheduler, _writer = await _dispatcher(SchedulerType.LOAD_BALANCE, decode_type)

    response = await _dispatch(dispatcher, SchedulerRequestType.TARS_STATUS, {}, "disabled")

    assert dispatcher._tars_activation is None  # pylint: disable=protected-access
    assert scheduler.get_tars_state_runtime() is None
    assert response.response_type == SchedulerResponseType.ERROR
    assert "Unknown request type" in (response.error or "")


@pytest.mark.parametrize(
    ("prefill", "decode"),
    [(SchedulerType.TARS, SchedulerType.LOAD_BALANCE), (SchedulerType.ROUND_ROBIN, SchedulerType.TARS)],
)
def test_scheduler_and_dispatcher_reject_unpaired_roles_before_runtime_creation(prefill, decode, monkeypatch):
    config = CoordinatorConfig()
    config.scheduler_config.prefill_scheduler_type = prefill
    config.scheduler_config.decode_scheduler_type = decode

    def unexpected_policy(*_args, **_kwargs):
        pytest.fail("Invalid TARS roles must be rejected before creating policies")

    monkeypatch.setattr("motor.coordinator.scheduler.scheduler.SchedulingPolicyFactory.create", unexpected_policy)
    with pytest.raises(ValueError, match="prefill_scheduler_type and decode_scheduler_type"):
        Scheduler(object(), config)
    with pytest.raises(ValueError, match="prefill_scheduler_type and decode_scheduler_type"):
        _SchedulerRequestDispatcher(object(), object(), config)


async def test_decision_log_is_compact_json_with_epoch_and_exact_target(monkeypatch):
    dispatcher, _manager, _scheduler, _writer = await _dispatcher(SchedulerType.TARS)
    activation = dispatcher._tars_activation  # pylint: disable=protected-access
    activation._config.scheduler_config.tars.enable_decision_log = True  # pylint: disable=protected-access
    messages: list[str] = []
    monkeypatch.setattr(
        activation_module.logger,
        "info",
        lambda message, *args: messages.append(message % args),
    )

    activation._decision(  # pylint: disable=protected-access
        "prepared",
        {"req_id": "request-1", "attempt_seq": 3},
        kind="single",
        target=("7", 1),
    )

    assert len(messages) == 1
    record = json.loads(messages[0].removeprefix("TARS decision "))
    assert record == {
        "attempt_seq": 3,
        "event": "prepared",
        "kind": "single",
        "req_id": "request-1",
        "scheduler_epoch": activation.scheduler_epoch,
        "target": ["7", 1],
    }


async def test_single_delivery_status_reconciles_confirm_and_normal_lifecycle(
    monkeypatch,
):
    dispatcher, manager, scheduler, writer = await _dispatcher(SchedulerType.TARS)
    await _add_topology(manager)
    activation = dispatcher._tars_activation  # pylint: disable=protected-access
    monkeypatch.setattr(
        activation,
        "_build_evidence",
        lambda instances, data: _evidence(instances, len(data["token_ids"])),
    )
    data = _admission_data()
    data["role"] = PDRole.ROLE_P.value

    prepared = await _dispatch(dispatcher, SchedulerRequestType.TARS_PREPARE_SINGLE, data, "prepare")
    assert prepared.response_type == SchedulerResponseType.SUCCESS
    delivery = prepared.data
    identity = delivery["identity"]
    delivery_data = {
        "kind": delivery["kind"],
        "req_id": identity["request_id"],
        "attempt_seq": identity["attempt_seq"],
        "scheduler_epoch": identity["scheduler_epoch"],
        "delivery_token": delivery["delivery_token"],
    }
    before = await _dispatch(dispatcher, SchedulerRequestType.TARS_STATUS, delivery_data, "status-1")
    confirmed = await _dispatch(dispatcher, SchedulerRequestType.TARS_CONFIRM, delivery_data, "confirm")
    after = await _dispatch(dispatcher, SchedulerRequestType.TARS_STATUS, delivery_data, "status-2")

    assert before.data["confirmed"] is False
    assert confirmed.response_type == SchedulerResponseType.SUCCESS
    assert after.data["confirmed"] is True
    assert writer.entries[(1, 0)]["active_tokens"] == 4

    lifecycle = dict(delivery_data, event="prefill_complete")
    assert (
        await _dispatch(dispatcher, SchedulerRequestType.TARS_LIFECYCLE, lifecycle, "prefill")
    ).response_type == SchedulerResponseType.SUCCESS
    decode = dict(delivery_data, instance_id=3, dp_rank=0)
    assert (
        await _dispatch(dispatcher, SchedulerRequestType.TARS_COMMIT_DECODE, decode, "decode")
    ).response_type == SchedulerResponseType.SUCCESS
    lifecycle["event"] = "request_complete"
    assert (
        await _dispatch(dispatcher, SchedulerRequestType.TARS_LIFECYCLE, lifecycle, "complete")
    ).response_type == SchedulerResponseType.SUCCESS
    assert scheduler.get_tars_state_runtime().snapshot().attempts == ()
    assert writer.entries[(1, 0)]["active_tokens"] == 0


async def test_joint_revoke_compensates_both_actual_workloads(monkeypatch):
    dispatcher, manager, scheduler, writer = await _dispatcher(SchedulerType.TARS)
    await _add_topology(manager)
    activation = dispatcher._tars_activation  # pylint: disable=protected-access
    monkeypatch.setattr(
        activation,
        "_build_evidence",
        lambda instances, data: _evidence(instances, len(data["token_ids"])),
    )

    prepared = await _dispatch(
        dispatcher,
        SchedulerRequestType.TARS_PREPARE_JOINT,
        _admission_data("bootstrap"),
        "joint",
    )
    assert prepared.response_type == SchedulerResponseType.SUCCESS
    delivery = prepared.data
    identity = delivery["identity"]
    delivery_data = {
        "kind": delivery["kind"],
        "req_id": identity["request_id"],
        "attempt_seq": identity["attempt_seq"],
        "scheduler_epoch": identity["scheduler_epoch"],
        "delivery_token": delivery["delivery_token"],
    }
    assert writer.entries[(3, 0)]["active_tokens"] == 8

    revoked = await _dispatch(dispatcher, SchedulerRequestType.TARS_REVOKE, delivery_data, "revoke")

    assert revoked.response_type == SchedulerResponseType.SUCCESS
    assert all(entry["active_tokens"] == 0 for entry in writer.entries.values())
    assert scheduler.get_tars_state_runtime().snapshot().attempts == ()
    assert len(writer.writes) == 4


async def test_unavailable_evidence_falls_back_without_tars_or_actual_mutation(
    monkeypatch,
):
    dispatcher, manager, scheduler, writer = await _dispatcher(SchedulerType.TARS)
    await _add_topology(manager)
    activation = dispatcher._tars_activation  # pylint: disable=protected-access
    monkeypatch.setattr(activation, "_build_evidence", lambda _instances, _data: None)
    data = _admission_data()
    data["role"] = PDRole.ROLE_P.value

    response = await _dispatch(dispatcher, SchedulerRequestType.TARS_PREPARE_SINGLE, data, "fallback")

    assert response.response_type == SchedulerResponseType.SUCCESS
    assert response.data == {"fallback": "load_balance"}
    assert scheduler.get_tars_state_runtime().snapshot().attempts == ()
    assert writer.writes == []
    assert all(
        endpoint.workload.active_tokens == 0
        for instance in manager.get_available_instances(None).values()
        for endpoint in instance.get_all_endpoints()
    )


async def test_drain_rejects_joint_admission_and_quiesce_reverses_confirmed_actual(
    monkeypatch,
):
    dispatcher, manager, scheduler, writer = await _dispatcher(SchedulerType.TARS)
    await _add_topology(manager)
    activation = dispatcher._tars_activation  # pylint: disable=protected-access
    monkeypatch.setattr(
        activation,
        "_build_evidence",
        lambda instances, data: _evidence(instances, len(data["token_ids"])),
    )
    prepared = await _dispatch(
        dispatcher,
        SchedulerRequestType.TARS_PREPARE_JOINT,
        _admission_data("bootstrap"),
        "joint-drain",
    )
    delivery = prepared.data
    identity = delivery["identity"]
    delivery_data = {
        "kind": delivery["kind"],
        "req_id": identity["request_id"],
        "attempt_seq": identity["attempt_seq"],
        "scheduler_epoch": identity["scheduler_epoch"],
        "delivery_token": delivery["delivery_token"],
    }
    assert (
        await _dispatch(
            dispatcher,
            SchedulerRequestType.TARS_CONFIRM,
            delivery_data,
            "confirm-drain",
        )
    ).response_type == SchedulerResponseType.SUCCESS

    dispatcher.begin_tars_drain()
    rejected_data = _admission_data("bootstrap")
    rejected_data.update({"req_id": "request-2", "attempt_seq": 2})
    rejected = await _dispatch(
        dispatcher,
        SchedulerRequestType.TARS_PREPARE_JOINT,
        rejected_data,
        "rejected-drain",
    )

    assert rejected.response_type == SchedulerResponseType.ERROR
    assert rejected.data["error_code"] == "server_draining"
    assert dispatcher.tars_drain_complete() is False

    dispatcher.quiesce_tars()

    assert all(entry["active_tokens"] == 0 for entry in writer.entries.values())
    assert scheduler.get_tars_state_runtime().snapshot().attempts == ()


def _delivery_data(delivery):
    return {
        "kind": delivery["kind"],
        "req_id": delivery["identity"]["request_id"],
        "attempt_seq": delivery["identity"]["attempt_seq"],
        "scheduler_epoch": delivery["identity"]["scheduler_epoch"],
        "delivery_token": delivery["delivery_token"],
    }


async def _prepared(monkeypatch, *, joint=False):
    dispatcher, manager, scheduler, writer = await _dispatcher(SchedulerType.TARS)
    await _add_topology(manager)
    monkeypatch.setattr(
        dispatcher._tars_activation,
        "_build_evidence",
        lambda instances, data: _evidence(instances, len(data["token_ids"])),
    )
    data = dict(_admission_data("bootstrap" if joint else "handoff"), role=PDRole.ROLE_P.value)
    request_type = SchedulerRequestType.TARS_PREPARE_JOINT if joint else SchedulerRequestType.TARS_PREPARE_SINGLE
    prepared = await _dispatch(dispatcher, request_type, data)
    assert prepared.response_type == SchedulerResponseType.SUCCESS
    return dispatcher, writer, data, request_type, prepared.data


@pytest.mark.parametrize("joint", [False, True])
async def test_duplicate_prepare_does_not_double_charge_shm(monkeypatch, joint):
    dispatcher, writer, data, request_type, delivery = await _prepared(monkeypatch, joint=joint)
    # Reconciliation must not depend on Conductor still being available or unchanged.
    monkeypatch.setattr(
        dispatcher._tars_activation,
        "_build_evidence",
        lambda *_args: pytest.fail("Existing delivery must not query evidence again"),
    )
    before = list(writer.writes)
    duplicate = await _dispatch(dispatcher, request_type, data, "duplicate")
    assert duplicate.data == delivery
    assert writer.writes == before


@pytest.mark.parametrize("joint", [False, True])
async def test_duplicate_prepare_with_changed_input_is_rejected(monkeypatch, joint):
    dispatcher, writer, data, request_type, _delivery = await _prepared(monkeypatch, joint=joint)
    before = list(writer.writes)
    response = await _dispatch(dispatcher, request_type, dict(data, token_ids=[5, 6]))
    assert response.data["error_code"] == "attempt_conflict"
    assert writer.writes == before


@pytest.mark.parametrize("joint", [False, True])
async def test_concurrent_prepare_retries_replay_first_commit_before_fallback(monkeypatch, joint):
    dispatcher, manager, _scheduler, writer = await _dispatcher(SchedulerType.TARS)
    await _add_topology(manager)
    entered = [asyncio.Event(), asyncio.Event()]
    resume = [threading.Event(), threading.Event()]
    sequence = count()
    loop = asyncio.get_running_loop()

    def evidence(instances, data):
        index = next(sequence)
        loop.call_soon_threadsafe(entered[index].set)
        assert resume[index].wait(timeout=10)
        return _evidence(instances, len(data["token_ids"])) if index == 0 else None

    monkeypatch.setattr(dispatcher._tars_activation, "_build_evidence", evidence)
    data = dict(_admission_data("bootstrap" if joint else "handoff"), role=PDRole.ROLE_P.value)
    request_type = SchedulerRequestType.TARS_PREPARE_JOINT if joint else SchedulerRequestType.TARS_PREPARE_SINGLE
    first = asyncio.create_task(_dispatch(dispatcher, request_type, data))
    second = None
    try:
        await asyncio.wait_for(entered[0].wait(), timeout=10)
        second = asyncio.create_task(_dispatch(dispatcher, request_type, data, "retry"))
        await asyncio.wait_for(entered[1].wait(), timeout=10)
        resume[0].set()
        original = await first
    finally:
        for barrier in resume:
            barrier.set()
    assert second is not None
    replay = await second
    assert original.response_type == SchedulerResponseType.SUCCESS
    assert replay.data == original.data
    assert len(writer.writes) == (2 if joint else 1)


@pytest.mark.parametrize("settle", ["release", "revoke", "terminal", "quiesce"])
async def test_late_release_does_not_debit_another_request(monkeypatch, settle):
    dispatcher, writer, _data, _request_type, delivery = await _prepared(monkeypatch)
    identity = _delivery_data(delivery)
    release = dict(
        identity,
        instance_id=delivery["instance"]["id"],
        endpoint_id=delivery["endpoint"]["id"],
        role=PDRole.ROLE_P.value,
    )
    if settle == "release":
        response = await _dispatch(dispatcher, SchedulerRequestType.TARS_RELEASE_WORKLOAD, release)
        assert response.data["success"]
    elif settle == "revoke":
        response = await _dispatch(dispatcher, SchedulerRequestType.TARS_REVOKE, identity)
        assert response.data["success"]
    elif settle == "terminal":
        await _dispatch(dispatcher, SchedulerRequestType.TARS_CONFIRM, identity)
        response = await _dispatch(dispatcher, SchedulerRequestType.TARS_LIFECYCLE, dict(identity, event="abort"))
        assert response.data["success"]
    else:
        await _dispatch(dispatcher, SchedulerRequestType.TARS_CONFIRM, identity)
        dispatcher.quiesce_tars()
    # Another ordinary worker request now owns nine tokens in the same slot.
    entry = writer.entries[(release["instance_id"], release["endpoint_id"])]
    assert entry["active_tokens"] == 0
    assert writer.cas_add(release["instance_id"], release["endpoint_id"], 1, 0.0, 9.0)[0] == STATUS_OK
    for _ in range(2):
        response = await _dispatch(dispatcher, SchedulerRequestType.TARS_RELEASE_WORKLOAD, release)
        assert response.data["success"]
    assert entry["active_tokens"] == 9
    assert entry["request_count"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"delivery_token": "wrong"},
        {"scheduler_epoch": 999},
        {"instance_id": 3},
        {"endpoint_id": 7},
        {"role": PDRole.ROLE_D.value},
        {"attempt_seq": 2},
    ],
)
async def test_release_rejects_foreign_receipt_without_mutation(monkeypatch, change):
    dispatcher, writer, _data, _request_type, delivery = await _prepared(monkeypatch)
    release = dict(
        _delivery_data(delivery),
        instance_id=delivery["instance"]["id"],
        endpoint_id=delivery["endpoint"]["id"],
        role=PDRole.ROLE_P.value,
    )
    release.update(change)
    before = list(writer.writes)
    response = await _dispatch(dispatcher, SchedulerRequestType.TARS_RELEASE_WORKLOAD, release)
    assert response.response_type == SchedulerResponseType.ERROR
    assert writer.writes == before


async def test_failed_terminal_compensation_is_retried_by_poll(monkeypatch):
    dispatcher, writer, _data, _request_type, delivery = await _prepared(monkeypatch, joint=True)
    identity = _delivery_data(delivery)
    await _dispatch(dispatcher, SchedulerRequestType.TARS_CONFIRM, identity)
    writer.fail_release = True
    response = await _dispatch(dispatcher, SchedulerRequestType.TARS_LIFECYCLE, dict(identity, event="abort"))
    assert response.response_type == SchedulerResponseType.ERROR
    assert sum(entry["active_tokens"] for entry in writer.entries.values()) == 12
    writer.fail_release = False
    dispatcher.poll_tars()
    assert all(entry["active_tokens"] == 0 for entry in writer.entries.values())


async def test_joint_decode_commit_failure_compensates_prefill(monkeypatch):
    dispatcher, manager, scheduler, writer = await _dispatcher(SchedulerType.TARS)
    await _add_topology(manager)
    monkeypatch.setattr(
        dispatcher._tars_activation,
        "_build_evidence",
        lambda instances, data: _evidence(instances, len(data["token_ids"])),
    )
    writer.blocked.add(3)
    response = await _dispatch(dispatcher, SchedulerRequestType.TARS_PREPARE_JOINT, _admission_data("bootstrap"))
    assert response.data["error_code"] == "workload_commit_failed"
    assert all(entry["active_tokens"] == 0 for entry in writer.entries.values())
    assert scheduler.get_tars_state_runtime().snapshot().attempts == ()


async def test_joint_evidence_in_flight_cannot_admit_after_drain(monkeypatch):
    dispatcher, manager, scheduler, writer = await _dispatcher(SchedulerType.TARS)
    await _add_topology(manager)
    entered = asyncio.Event()
    resume = threading.Event()
    loop = asyncio.get_running_loop()

    def evidence(instances, data):
        loop.call_soon_threadsafe(entered.set)
        assert resume.wait(timeout=5)
        return _evidence(instances, len(data["token_ids"]))

    monkeypatch.setattr(dispatcher._tars_activation, "_build_evidence", evidence)
    task = asyncio.create_task(
        _dispatch(dispatcher, SchedulerRequestType.TARS_PREPARE_JOINT, _admission_data("bootstrap"))
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        dispatcher.begin_tars_drain()
    finally:
        resume.set()
    response = await task
    assert response.data["error_code"] == "server_draining"
    assert writer.writes == []
    assert scheduler.get_tars_state_runtime().snapshot().attempts == ()


async def test_client_dispatcher_joint_roundtrip_reconciles_lost_release_ack(monkeypatch):
    dispatcher, manager, _scheduler, writer = await _dispatcher(SchedulerType.TARS)
    await _add_topology(manager)
    monkeypatch.setattr(
        dispatcher._tars_activation,
        "_build_evidence",
        lambda instances, data: _evidence(instances, len(data["token_ids"])),
    )
    client = AsyncSchedulerClient(SchedulerClientConfig(scheduler_type="tars"))
    lose_release_ack = True

    async def transport(request):
        nonlocal lose_release_ack
        response = await dispatcher.dispatch(request)
        if request.request_type == SchedulerRequestType.TARS_RELEASE_WORKLOAD and lose_release_ack:
            lose_release_ack = False
            return None
        return response

    monkeypatch.setattr(client._transport, "send_request", transport)
    request_info = RequestInfo(
        req_id="roundtrip", req_data={"max_tokens": 8}, req_len=4, token_ids=[1, 2, 3, 4], api="v1/chat/completions"
    )
    pair = await client.tars_allocate_bootstrap_joint(request_info, attempt_seq=1)
    assert pair is not None
    assert sum(entry["request_count"] for entry in writer.entries.values()) == 2
    assert await client.tars_abort("roundtrip", 1)
    assert all(entry["active_tokens"] == 0 for entry in writer.entries.values())
    for instance, endpoint, workload in pair:
        entry = writer.entries[(instance.id, endpoint.id)]
        writer.cas_add(instance.id, endpoint.id, entry["generation"], 0.0, 11.0)
        params = UpdateWorkloadParams(
            instance_id=instance.id,
            endpoint_id=endpoint.id,
            role=PDRole(instance.role),
            req_id="roundtrip",
            attempt_seq=1,
            workload_action=WorkloadAction.RELEASE_TOKENS,
            workload_change=Workload(active_tokens=-workload.active_tokens),
        )
        if lose_release_ack:
            assert await client.update_workload(params) is False
        assert await client.update_workload(params) is True
        assert await client.update_workload(params) is True
        assert entry["active_tokens"] == 11
        assert entry["request_count"] == 1
