# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Protocol invariants for the dormant TARS Handoff single-role path."""

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from motor.coordinator.scheduler.policy.tars.handoff_protocol import (
    HandoffAllocateRequest,
    HandoffAttemptId,
    HandoffAttemptSequencer,
    HandoffAttemptState,
    HandoffBoundaryFailure,
    HandoffConfirmRequest,
    HandoffDecodeCommitRequest,
    HandoffFailureCode,
    HandoffFaultPoint,
    HandoffLifecycleEvent,
    HandoffLifecycleRequest,
    HandoffProtocolClient,
    HandoffProtocolServer,
    HandoffRecoveryEvidence,
    HandoffRevokeRequest,
    HandoffRole,
    HandoffRouterLedger,
    HandoffServerState,
    HandoffStartupMode,
    HandoffStatusRequest,
    HandoffTargetSpec,
    HandoffTimeoutBudget,
)
from motor.coordinator.scheduler.policy.tars.models import (
    CacheCreditCapability,
    TarsCacheEvidence,
)
from motor.coordinator.scheduler.policy.tars.original.models import (
    TarsOriginalConfig,
    TarsRequestContext,
    TarsTarget,
)
from motor.coordinator.scheduler.policy.tars.state_runtime import (
    TarsAttemptKey,
    TarsAttemptSpec,
    TarsReplicaSpec,
    TarsStateRuntime,
)


@dataclass
class _Harness:
    runtime: TarsStateRuntime
    server: HandoffProtocolServer
    ledger: HandoffRouterLedger
    client: HandoffProtocolClient
    prefill_targets: tuple[TarsTarget, ...]
    decode_targets: tuple[TarsTarget, ...]


class _FailOnce:
    def __init__(self, fault_point: HandoffFaultPoint) -> None:
        self.fault_point = fault_point
        self.failed: set[HandoffAttemptId] = set()

    def __call__(self, fault_point: HandoffFaultPoint, identity: HandoffAttemptId) -> None:
        if fault_point is self.fault_point and identity not in self.failed:
            self.failed.add(identity)
            raise HandoffBoundaryFailure(f"injected at {fault_point.value}")


@dataclass
class _Clock:
    value: float = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _RaceHook:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.action = "success"
        self.cancellation: asyncio.CancelledError | None = None

    def __call__(self, fault_point: HandoffFaultPoint, identity: HandoffAttemptId) -> None:
        del identity
        if self.action == "server_timeout" and fault_point is HandoffFaultPoint.AFTER_WORKLOAD_COMMIT:
            self.clock.advance(1.1)
        elif self.action == "client_timeout" and fault_point is HandoffFaultPoint.AFTER_ROUTER_PERSIST:
            self.clock.advance(5.1)
        elif self.action == "cancel" and fault_point is HandoffFaultPoint.AFTER_ROUTER_PERSIST:
            self.cancellation = asyncio.CancelledError("caller cancelled allocation")
            raise self.cancellation


class _CancelOnce:
    def __init__(self, fault_point: HandoffFaultPoint) -> None:
        self.fault_point = fault_point
        self.cancellation = asyncio.CancelledError(f"cancelled at {fault_point.value}")

    def __call__(self, fault_point: HandoffFaultPoint, identity: HandoffAttemptId) -> None:
        del identity
        if fault_point is self.fault_point:
            raise self.cancellation


def _target(instance_id: int, dp_rank: int = 0) -> TarsTarget:
    return TarsTarget(instance_id=str(instance_id), dp_rank=dp_rank)


def _evidence(target: TarsTarget, input_tokens: int, credit_tokens: int) -> TarsCacheEvidence:
    return TarsCacheEvidence(
        target=target,
        input_tokens=input_tokens,
        capability=CacheCreditCapability.DP_NPU_ONLY,
        observed_exact_dp_npu_tokens=credit_tokens,
        best_effort_cache_credit_tokens=credit_tokens,
        estimated_compute_tokens=input_tokens - credit_tokens,
        diagnostic_matched_tokens=credit_tokens,
        diagnostic_cpu_blocks=0,
        diagnostic_disk_blocks=0,
    )


def _request(
    identity: HandoffAttemptId,
    prefill_targets: tuple[TarsTarget, ...],
    *,
    scheduler_epoch: int | None = None,
    topology_epoch: int = 1,
    input_tokens: int = 8,
) -> HandoffAllocateRequest:
    cache_credits = tuple(input_tokens - index - 1 for index, _ in enumerate(prefill_targets))
    spec = TarsAttemptSpec(
        key=TarsAttemptKey(identity.request_id, identity.attempt_seq),
        request=TarsRequestContext(
            session_id=f"session-{identity.request_id}",
            request_id=identity.request_id,
            input_tokens=input_tokens,
        ),
        evidence=tuple(
            _evidence(target, input_tokens, credit)
            for target, credit in zip(prefill_targets, cache_credits, strict=True)
        ),
        max_output_tokens=4,
        block_size=4,
    )
    return HandoffAllocateRequest(
        identity=identity,
        attempt_spec=spec,
        scheduler_epoch=identity.scheduler_epoch if scheduler_epoch is None else scheduler_epoch,
        expected_topology_epoch=topology_epoch,
    )


def _target_specs(
    prefill_targets: tuple[TarsTarget, ...],
    decode_targets: tuple[TarsTarget, ...],
    *,
    circuit_open: TarsTarget | None = None,
    decode_engine_type: str = "vllm",
) -> tuple[HandoffTargetSpec, ...]:
    return tuple(
        HandoffTargetSpec(target=target, role=HandoffRole.PREFILL, engine_type="vllm") for target in prefill_targets
    ) + tuple(
        HandoffTargetSpec(
            target=target,
            role=HandoffRole.DECODE,
            engine_type=decode_engine_type,
            circuit_open=target == circuit_open,
        )
        for target in decode_targets
    )


def _harness(
    fault_hook: Callable[[HandoffFaultPoint, HandoffAttemptId], None] | None = None,
    *,
    clock: _Clock | None = None,
    timeout_budget: HandoffTimeoutBudget | None = None,
) -> _Harness:
    prefill_targets = (_target(100, 0), _target(100, 1))
    decode_targets = (_target(200, 0), _target(200, 1))
    all_targets = prefill_targets + decode_targets
    runtime = TarsStateRuntime(
        scheduler_id="scheduler-handoff",
        config=TarsOriginalConfig(tpct=1.0, tprt=0.0, dh_first_balance_ttft_threshold=100),
        clock=clock,
    )
    runtime.sync_topology(
        1,
        tuple(
            TarsReplicaSpec(
                target=target,
                pending_token_capacity=128,
                running_block_capacity=128,
            )
            for target in all_targets
        ),
    )
    server = HandoffProtocolServer(
        runtime=runtime,
        scheduler_epoch=7,
        fault_hook=fault_hook,
        clock=clock,
        timeout_budget=timeout_budget,
    )
    assert server.sync_targets(1, _target_specs(prefill_targets, decode_targets))
    ledger = HandoffRouterLedger()
    client = HandoffProtocolClient(
        server=server,
        router_ledger=ledger,
        fault_hook=fault_hook,
        clock=clock,
        timeout_budget=timeout_budget,
    )
    return _Harness(runtime, server, ledger, client, prefill_targets, decode_targets)


def _assert_no_transient_state(harness: _Harness) -> None:
    server = harness.server.snapshot()
    runtime = harness.runtime.snapshot()
    router = harness.ledger.snapshot()
    assert server.active_attempts == ()
    assert server.pending_deliveries == ()
    assert server.decode_workloads == ()
    assert runtime.attempts == ()
    assert runtime.queue_entries == ()
    assert runtime.reservations == ()
    assert runtime.workloads == ()
    assert router.entries == ()


def test_handoff_confirmation_gates_dispatch_and_decode_remains_deferred() -> None:
    harness = _harness()
    identity = HandoffAttemptId("request-confirm", 1, 7)
    request = _request(identity, harness.prefill_targets)

    allocated = harness.server.allocate(request)
    assert allocated.ok
    assert allocated.allocation is not None
    assert not allocated.allocation.confirmed
    assert allocated.attempt is not None
    assert allocated.attempt.state is HandoffAttemptState.DELIVERY_PENDING
    assert harness.ledger.persist(allocated.allocation).ok

    preconfirm_dispatch = harness.ledger.mark_dispatched(identity)
    assert not preconfirm_dispatch.ok
    assert preconfirm_dispatch.failure is not None
    assert preconfirm_dispatch.failure.code is HandoffFailureCode.PRECONFIRM_DISPATCH

    early_decode = harness.server.commit_decode(HandoffDecodeCommitRequest(identity, harness.decode_targets[0], 7, 1))
    assert not early_decode.ok
    assert early_decode.failure is not None
    assert early_decode.failure.code is HandoffFailureCode.INVALID_STATE

    wrong_token = harness.server.confirm(HandoffConfirmRequest(identity, "wrong-token", 7))
    assert not wrong_token.ok
    assert wrong_token.failure is not None
    assert wrong_token.failure.code is HandoffFailureCode.DELIVERY_TOKEN_MISMATCH

    confirmed = harness.server.confirm(HandoffConfirmRequest(identity, allocated.allocation.delivery_token, 7))
    assert confirmed.ok
    assert confirmed.allocation is not None
    assert confirmed.allocation.confirmed
    assert harness.ledger.confirm(confirmed.allocation).ok
    assert harness.ledger.mark_dispatched(identity).ok

    settled = harness.server.lifecycle(
        HandoffLifecycleRequest(
            identity,
            HandoffLifecycleEvent.PREFILL_COMPLETE,
            7,
            observed_ttft=1.25,
        )
    )
    assert settled.ok
    assert settled.attempt is not None
    assert settled.attempt.state is HandoffAttemptState.PREFILL_COMPLETED
    assert harness.server.commit_decode(HandoffDecodeCommitRequest(identity, harness.decode_targets[0], 7, 1)).ok
    completed = harness.server.lifecycle(HandoffLifecycleRequest(identity, HandoffLifecycleEvent.REQUEST_COMPLETE, 7))
    assert completed.ok
    assert completed.attempt is not None
    assert completed.attempt.state is HandoffAttemptState.REQUEST_COMPLETED
    assert harness.ledger.remove(identity).ok
    assert harness.server.lifecycle(HandoffLifecycleRequest(identity, HandoffLifecycleEvent.REQUEST_COMPLETE, 7)).ok

    _assert_no_transient_state(harness)
    harness.client.quiesce()
    assert harness.server.snapshot().terminal_receipts == ()
    assert len(harness.server.snapshot().target_inventory) == 4


def test_protocol_returns_typed_failures_and_idempotent_compensation() -> None:
    harness = _harness()
    identity = HandoffAttemptId("request-errors", 1, 7)
    stale = harness.server.allocate(_request(identity, harness.prefill_targets, scheduler_epoch=6))
    assert not stale.ok
    assert stale.failure is not None
    assert stale.failure.code is HandoffFailureCode.STALE_SCHEDULER_EPOCH

    request = _request(identity, harness.prefill_targets)
    first = harness.server.allocate(request)
    second = harness.server.allocate(request)
    assert first.ok and second.ok
    assert first.allocation == second.allocation
    assert first.allocation is not None

    conflicting = harness.server.allocate(_request(identity, harness.prefill_targets, input_tokens=9))
    assert not conflicting.ok
    assert conflicting.failure is not None
    assert conflicting.failure.code is HandoffFailureCode.ATTEMPT_CONFLICT

    wrong_revoke = harness.server.revoke(HandoffRevokeRequest(identity, "wrong-token", 7))
    assert not wrong_revoke.ok
    assert wrong_revoke.failure is not None
    assert wrong_revoke.failure.code is HandoffFailureCode.DELIVERY_TOKEN_MISMATCH
    revoked = harness.server.revoke(HandoffRevokeRequest(identity, first.allocation.delivery_token, 7))
    assert revoked.ok
    assert revoked.attempt is not None
    assert revoked.attempt.state is HandoffAttemptState.ALLOCATION_REVOKED
    assert harness.server.revoke(HandoffRevokeRequest(identity, first.allocation.delivery_token, 7)).ok
    terminal_wrong_token = harness.server.revoke(HandoffRevokeRequest(identity, "wrong-token", 7))
    assert not terminal_wrong_token.ok
    assert terminal_wrong_token.failure is not None
    assert terminal_wrong_token.failure.code is HandoffFailureCode.DELIVERY_TOKEN_MISMATCH

    missing = harness.server.status(HandoffStatusRequest(HandoffAttemptId("missing", 1, 7), 7))
    assert not missing.ok
    assert missing.failure is not None
    assert missing.failure.code is HandoffFailureCode.ATTEMPT_NOT_FOUND
    _assert_no_transient_state(harness)


def test_prefill_settlement_and_decode_guards_are_idempotent() -> None:
    hook = _FailOnce(HandoffFaultPoint.AFTER_PREFILL_RELEASE)
    harness = _harness(hook)
    identity = HandoffAttemptId("request-settle", 1, 7)
    allocated = harness.client.allocate(_request(identity, harness.prefill_targets))
    assert allocated.ok
    retried = harness.client.allocate(_request(identity, harness.prefill_targets))
    assert retried.ok
    assert retried.allocation == allocated.allocation

    first_settle = harness.client.lifecycle(
        HandoffLifecycleRequest(
            identity,
            HandoffLifecycleEvent.PREFILL_COMPLETE,
            7,
            observed_ttft=0.8,
        )
    )
    assert not first_settle.ok
    assert first_settle.failure is not None
    assert first_settle.failure.code is HandoffFailureCode.BOUNDARY_FAILURE
    retry_settle = harness.client.lifecycle(
        HandoffLifecycleRequest(
            identity,
            HandoffLifecycleEvent.PREFILL_COMPLETE,
            7,
            observed_ttft=0.8,
        )
    )
    assert retry_settle.ok
    assert harness.client.lifecycle(
        HandoffLifecycleRequest(
            identity,
            HandoffLifecycleEvent.PREFILL_COMPLETE,
            7,
            observed_ttft=0.8,
        )
    ).ok

    stale_topology = harness.client.commit_decode(HandoffDecodeCommitRequest(identity, harness.decode_targets[0], 7, 0))
    assert not stale_topology.ok
    assert stale_topology.failure is not None
    assert stale_topology.failure.code is HandoffFailureCode.STALE_TOPOLOGY

    harness.server.sync_targets(
        1,
        _target_specs(
            harness.prefill_targets,
            harness.decode_targets,
            circuit_open=harness.decode_targets[0],
        ),
    )
    circuit_open = harness.client.commit_decode(HandoffDecodeCommitRequest(identity, harness.decode_targets[0], 7, 1))
    assert not circuit_open.ok
    assert circuit_open.failure is not None
    assert circuit_open.failure.code is HandoffFailureCode.CIRCUIT_OPEN

    harness.server.sync_targets(
        1,
        _target_specs(
            harness.prefill_targets,
            harness.decode_targets,
            decode_engine_type="sglang",
        ),
    )
    engine_mismatch = harness.client.commit_decode(
        HandoffDecodeCommitRequest(identity, harness.decode_targets[0], 7, 1)
    )
    assert not engine_mismatch.ok
    assert engine_mismatch.failure is not None
    assert engine_mismatch.failure.code is HandoffFailureCode.ENGINE_TYPE_MISMATCH

    aborted = harness.client.lifecycle(HandoffLifecycleRequest(identity, HandoffLifecycleEvent.ABORT, 7))
    assert aborted.ok
    assert harness.client.lifecycle(HandoffLifecycleRequest(identity, HandoffLifecycleEvent.ABORT, 7)).ok
    _assert_no_transient_state(harness)


def test_each_protocol_fault_boundary_converges_100_times() -> None:
    allocation_faults = {
        HandoffFaultPoint.AFTER_ENQUEUE,
        HandoffFaultPoint.AFTER_TARGET_RESERVATION,
        HandoffFaultPoint.AFTER_WORKLOAD_COMMIT,
        HandoffFaultPoint.AFTER_DELIVERY_PREPARE,
        HandoffFaultPoint.AFTER_ROUTER_PERSIST,
        HandoffFaultPoint.AFTER_CONFIRM,
    }
    fault_points = allocation_faults | {
        HandoffFaultPoint.AFTER_PREFILL_RELEASE,
        HandoffFaultPoint.BEFORE_DECODE_COMMIT,
    }

    for fault_point in fault_points:
        hook = _FailOnce(fault_point)
        harness = _harness(hook)
        for request_number in range(100):
            identity = HandoffAttemptId(f"fault-{fault_point.value}-{request_number}", 1, 7)
            allocated = harness.client.allocate(_request(identity, harness.prefill_targets))
            if fault_point in allocation_faults:
                assert not allocated.ok
            else:
                assert allocated.ok
                settled = harness.client.lifecycle(
                    HandoffLifecycleRequest(
                        identity,
                        HandoffLifecycleEvent.PREFILL_COMPLETE,
                        7,
                        observed_ttft=1.0,
                    )
                )
                if fault_point is HandoffFaultPoint.AFTER_PREFILL_RELEASE:
                    assert not settled.ok
                    assert harness.client.lifecycle(
                        HandoffLifecycleRequest(
                            identity,
                            HandoffLifecycleEvent.PREFILL_COMPLETE,
                            7,
                            observed_ttft=1.0,
                        )
                    ).ok
                    assert harness.client.commit_decode(
                        HandoffDecodeCommitRequest(identity, harness.decode_targets[0], 7, 1)
                    ).ok
                    assert harness.client.lifecycle(
                        HandoffLifecycleRequest(identity, HandoffLifecycleEvent.REQUEST_COMPLETE, 7)
                    ).ok
                else:
                    assert settled.ok
                    decode = harness.client.commit_decode(
                        HandoffDecodeCommitRequest(identity, harness.decode_targets[0], 7, 1)
                    )
                    assert not decode.ok
                    assert harness.client.lifecycle(
                        HandoffLifecycleRequest(identity, HandoffLifecycleEvent.ABORT, 7)
                    ).ok
            harness.server.assert_invariants()
            _assert_no_transient_state(harness)

        assert len(hook.failed) == 100
        harness.client.quiesce()
        assert harness.server.snapshot().terminal_receipts == ()


def test_1000_handoff_requests_preserve_identity_and_all_ledgers_converge() -> None:
    harness = _harness()
    sequencer = HandoffAttemptSequencer(root_request_id="request-1000", scheduler_epoch=7)
    attempt_sequences: list[int] = []

    for request_number in range(1_000):
        identity = sequencer.reserve()
        attempt_sequences.append(identity.attempt_seq)
        result = harness.client.allocate(_request(identity, harness.prefill_targets))
        assert result.ok
        assert result.allocation is not None
        allocation = result.allocation
        assert allocation.confirmed
        assert allocation.identity == identity

        status = harness.server.status(HandoffStatusRequest(identity, 7))
        assert status.ok
        assert status.attempt is not None
        assert status.attempt.target == allocation.target
        runtime = harness.runtime.snapshot()
        assert runtime.reservations == ((identity.key, allocation.target),)
        assert len(runtime.workloads) == 1
        assert runtime.workloads[0].target == allocation.target
        router_entry = harness.ledger.snapshot().entries
        assert len(router_entry) == 1
        assert router_entry[0].identity == identity
        assert router_entry[0].target == allocation.target
        assert router_entry[0].confirmed
        assert not router_entry[0].dispatched

        assert harness.client.dispatch(identity).ok
        assert (
            harness.client.commit_decode(
                HandoffDecodeCommitRequest(identity, harness.decode_targets[request_number % 2], 7, 1)
            ).failure.code
            is HandoffFailureCode.INVALID_STATE
        )
        assert harness.client.lifecycle(
            HandoffLifecycleRequest(
                identity,
                HandoffLifecycleEvent.PREFILL_COMPLETE,
                7,
                observed_ttft=1.0,
            )
        ).ok
        assert harness.client.commit_decode(
            HandoffDecodeCommitRequest(identity, harness.decode_targets[request_number % 2], 7, 1)
        ).ok
        assert harness.client.lifecycle(HandoffLifecycleRequest(identity, HandoffLifecycleEvent.REQUEST_COMPLETE, 7)).ok
        _assert_no_transient_state(harness)

    assert attempt_sequences == list(range(1, 1_001))
    assert len(set(attempt_sequences)) == 1_000
    assert len(harness.server.snapshot().terminal_receipts) == 1_000
    harness.client.quiesce()
    final_server = harness.server.snapshot()
    final_runtime = harness.runtime.snapshot()
    assert final_server.active_attempts == ()
    assert final_server.pending_deliveries == ()
    assert final_server.decode_workloads == ()
    assert final_server.terminal_receipts == ()
    assert len(final_server.target_inventory) == 4
    assert final_runtime.attempts == ()
    assert len(final_runtime.replicas) == 4


def test_timeout_budget_requires_strict_client_server_stage_ordering() -> None:
    valid = HandoffTimeoutBudget(
        queue_wait_s=0.25,
        final_query_s=0.25,
        exact_commit_s=0.25,
        settlement_margin_s=0.25,
        server_timeout_s=1.1,
        client_timeout_s=1.2,
        delivery_lease_s=2.0,
        drain_timeout_s=3.0,
        recovery_cooldown_s=4.0,
    )
    assert valid.client_timeout_s > valid.server_timeout_s
    with pytest.raises(ValueError, match="server_timeout_s"):
        HandoffTimeoutBudget(
            queue_wait_s=0.25,
            final_query_s=0.25,
            exact_commit_s=0.25,
            settlement_margin_s=0.25,
            server_timeout_s=1.0,
            client_timeout_s=1.2,
        )
    with pytest.raises(ValueError, match="client_timeout_s"):
        HandoffTimeoutBudget(server_timeout_s=5.0, client_timeout_s=5.0)


def test_delivery_lease_expiry_compensates_scheduler_and_reports_status() -> None:
    clock = _Clock()
    budget = HandoffTimeoutBudget(delivery_lease_s=2.0)
    harness = _harness(clock=clock, timeout_budget=budget)
    identity = HandoffAttemptId("lease-expiry", 1, 7)
    allocated = harness.server.allocate(_request(identity, harness.prefill_targets))
    assert allocated.ok and allocated.allocation is not None
    assert allocated.allocation.delivery_expires_at == pytest.approx(2.0)
    assert harness.ledger.persist(allocated.allocation).ok

    clock.advance(2.0)
    status = harness.server.status(HandoffStatusRequest(identity, 7))
    assert not status.ok
    assert status.failure is not None
    assert status.failure.code is HandoffFailureCode.DELIVERY_LEASE_EXPIRED
    late_confirm = harness.server.confirm(HandoffConfirmRequest(identity, allocated.allocation.delivery_token, 7))
    assert not late_confirm.ok
    assert late_confirm.failure == status.failure
    harness.client.poll()
    _assert_no_transient_state(harness)


def test_drain_rejects_admission_and_stops_after_deadline_without_leaks() -> None:
    clock = _Clock()
    harness = _harness(clock=clock)
    identity = HandoffAttemptId("drain-active", 1, 7)
    assert harness.client.allocate(_request(identity, harness.prefill_targets)).ok
    assert harness.server.begin_drain(timeout_s=2.0)
    assert harness.server.snapshot().state is HandoffServerState.DRAINING

    rejected = harness.client.allocate(_request(HandoffAttemptId("drain-rejected", 1, 7), harness.prefill_targets))
    assert not rejected.ok
    assert rejected.failure is not None
    assert rejected.failure.code is HandoffFailureCode.SERVER_DRAINING
    clock.advance(2.0)
    assert harness.client.poll() is HandoffServerState.STOPPED
    _assert_no_transient_state(harness)

    restarted = HandoffProtocolServer(
        runtime=harness.runtime,
        scheduler_epoch=8,
        clock=clock,
        startup_mode=HandoffStartupMode.GRACEFUL,
    )
    assert restarted.sync_targets(1, _target_specs(harness.prefill_targets, harness.decode_targets))
    restarted_client = HandoffProtocolClient(
        restarted,
        HandoffRouterLedger(),
        clock=clock,
    )
    restarted_identity = HandoffAttemptId("graceful-restart", 1, 8)
    assert restarted_client.allocate(_request(restarted_identity, harness.prefill_targets)).ok
    assert restarted_client.lifecycle(HandoffLifecycleRequest(restarted_identity, HandoffLifecycleEvent.ABORT, 8)).ok


def test_50_ungraceful_restart_cutpoints_require_explicit_reconciliation() -> None:
    for cutpoint in range(50):
        clock = _Clock()
        harness = _harness(clock=clock)
        identity = HandoffAttemptId(f"restart-{cutpoint}", 1, 7)
        if cutpoint % 3 == 0:
            allocated = harness.server.allocate(_request(identity, harness.prefill_targets))
            assert allocated.ok and allocated.allocation is not None
            assert harness.ledger.persist(allocated.allocation).ok
        else:
            allocated = harness.client.allocate(_request(identity, harness.prefill_targets))
            assert allocated.ok and allocated.allocation is not None
            if cutpoint % 3 == 2:
                assert harness.client.dispatch(identity).ok

        restarted = HandoffProtocolServer(
            runtime=harness.runtime,
            scheduler_epoch=8,
            clock=clock,
            startup_mode=HandoffStartupMode.UNGRACEFUL,
        )
        assert restarted.sync_targets(
            1,
            _target_specs(harness.prefill_targets, harness.decode_targets),
        )
        assert restarted.snapshot().state is HandoffServerState.RECOVERING
        rejected = restarted.allocate(
            _request(HandoffAttemptId(f"not-ready-{cutpoint}", 1, 8), harness.prefill_targets)
        )
        assert not rejected.ok
        assert rejected.failure is not None
        assert rejected.failure.code is HandoffFailureCode.SERVER_NOT_READY
        stale = restarted.confirm(HandoffConfirmRequest(identity, allocated.allocation.delivery_token, 7))
        assert not stale.ok
        assert stale.failure is not None
        assert stale.failure.code is HandoffFailureCode.STALE_SCHEDULER_EPOCH

        clock.advance(30.0)
        assert restarted.poll() is HandoffServerState.RECOVERING
        evidence = HandoffRecoveryEvidence(
            previous_scheduler_epoch=7,
            router_entries=tuple(entry.identity for entry in harness.ledger.snapshot().entries),
            operator_confirmed=True,
        )
        assert not restarted.complete_recovery(evidence).ok
        harness.ledger.quiesce()
        not_confirmed = restarted.complete_recovery(
            HandoffRecoveryEvidence(
                previous_scheduler_epoch=7,
                router_entries=(),
                operator_confirmed=False,
            )
        )
        assert not not_confirmed.ok
        accepted = restarted.complete_recovery(
            HandoffRecoveryEvidence(
                previous_scheduler_epoch=7,
                router_entries=(),
                operator_confirmed=True,
            )
        )
        assert accepted.ok
        assert restarted.snapshot().state is HandoffServerState.RUNNING
        assert harness.runtime.snapshot().attempts == ()
        assert harness.runtime.snapshot().reservations == ()
        assert harness.runtime.snapshot().workloads == ()


def test_10000_timeout_cancellation_races_converge_without_duplicate_dispatch() -> None:
    rng = random.Random(20260829)
    actions = ["success", "server_timeout", "client_timeout", "cancel"] * 25
    dispatched: set[HandoffAttemptId] = set()
    attempts = 0
    for batch in range(100):
        clock = _Clock()
        hook = _RaceHook(clock)
        budget = HandoffTimeoutBudget(
            queue_wait_s=0.25,
            final_query_s=0.25,
            exact_commit_s=0.25,
            settlement_margin_s=0.25,
            server_timeout_s=1.1,
            client_timeout_s=5.0,
        )
        harness = _harness(hook, clock=clock, timeout_budget=budget)
        rng.shuffle(actions)
        for item, action in enumerate(actions):
            attempts += 1
            identity = HandoffAttemptId(f"race-{batch}-{item}", 1, 7)
            hook.action = action
            if action == "cancel":
                with pytest.raises(asyncio.CancelledError) as cancelled:
                    harness.client.allocate(_request(identity, harness.prefill_targets))
                assert cancelled.value is hook.cancellation
            else:
                result = harness.client.allocate(_request(identity, harness.prefill_targets))
                if action == "success":
                    assert result.ok
                    assert identity not in dispatched
                    assert harness.client.dispatch(identity).ok
                    dispatched.add(identity)
                    assert harness.client.lifecycle(
                        HandoffLifecycleRequest(identity, HandoffLifecycleEvent.ABORT, 7)
                    ).ok
                else:
                    assert not result.ok
                    assert result.failure is not None
                    expected = (
                        HandoffFailureCode.SERVER_TIMEOUT
                        if action == "server_timeout"
                        else HandoffFailureCode.CLIENT_TIMEOUT
                    )
                    assert result.failure.code is expected
            harness.server.assert_invariants()
            _assert_no_transient_state(harness)
        harness.client.quiesce()

    assert attempts == 10_000
    assert len(dispatched) == 2_500


def test_cancellation_at_every_protocol_boundary_preserves_original_exception() -> None:
    allocation_faults = {
        HandoffFaultPoint.AFTER_ENQUEUE,
        HandoffFaultPoint.AFTER_TARGET_RESERVATION,
        HandoffFaultPoint.AFTER_WORKLOAD_COMMIT,
        HandoffFaultPoint.AFTER_DELIVERY_PREPARE,
        HandoffFaultPoint.AFTER_ROUTER_PERSIST,
        HandoffFaultPoint.AFTER_CONFIRM,
    }
    for fault_point in allocation_faults | {
        HandoffFaultPoint.AFTER_PREFILL_RELEASE,
        HandoffFaultPoint.BEFORE_DECODE_COMMIT,
    }:
        hook = _CancelOnce(fault_point)
        harness = _harness(hook)
        identity = HandoffAttemptId(f"cancel-{fault_point.value}", 1, 7)
        if fault_point in allocation_faults:
            with pytest.raises(asyncio.CancelledError) as cancelled:
                harness.client.allocate(_request(identity, harness.prefill_targets))
        else:
            assert harness.client.allocate(_request(identity, harness.prefill_targets)).ok
            if fault_point is HandoffFaultPoint.AFTER_PREFILL_RELEASE:
                with pytest.raises(asyncio.CancelledError) as cancelled:
                    harness.client.lifecycle(
                        HandoffLifecycleRequest(
                            identity,
                            HandoffLifecycleEvent.PREFILL_COMPLETE,
                            7,
                            observed_ttft=1.0,
                        )
                    )
            else:
                assert harness.client.lifecycle(
                    HandoffLifecycleRequest(
                        identity,
                        HandoffLifecycleEvent.PREFILL_COMPLETE,
                        7,
                        observed_ttft=1.0,
                    )
                ).ok
                with pytest.raises(asyncio.CancelledError) as cancelled:
                    harness.client.commit_decode(HandoffDecodeCommitRequest(identity, harness.decode_targets[0], 7, 1))
        assert cancelled.value is hook.cancellation
        _assert_no_transient_state(harness)
