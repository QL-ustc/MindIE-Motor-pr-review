# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Timing and isolation invariants for dormant Trigger and PDHybrid TARS adapters."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import pytest

from motor.coordinator.scheduler.policy.tars.handoff_protocol import (
    HandoffAllocateRequest,
    HandoffAllocation,
    HandoffAttemptId,
    HandoffAttemptSequencer,
    HandoffFailureCode,
    HandoffLifecycleEvent,
    HandoffLifecycleRequest,
    HandoffProtocolClient,
    HandoffProtocolServer,
    HandoffRole,
    HandoffRouterLedger,
    HandoffTargetSpec,
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
from motor.coordinator.scheduler.policy.tars.single_role_adapters import (
    PDHybridTarsAdapter,
    TriggerTarsAdapter,
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


def _target(instance_id: int, dp_rank: int = 0) -> TarsTarget:
    return TarsTarget(instance_id=str(instance_id), dp_rank=dp_rank)


def _fixed_clock() -> float:
    return 0.0


def _evidence(target: TarsTarget, input_tokens: int, cache_credit: int) -> TarsCacheEvidence:
    return TarsCacheEvidence(
        target=target,
        input_tokens=input_tokens,
        capability=CacheCreditCapability.DP_NPU_ONLY,
        observed_exact_dp_npu_tokens=cache_credit,
        best_effort_cache_credit_tokens=cache_credit,
        estimated_compute_tokens=input_tokens - cache_credit,
        diagnostic_matched_tokens=cache_credit,
        diagnostic_cpu_blocks=0,
        diagnostic_disk_blocks=0,
    )


def _request(
    identity: HandoffAttemptId,
    targets: tuple[TarsTarget, ...],
    role: HandoffRole,
    *,
    topology_epoch: int = 1,
) -> HandoffAllocateRequest:
    input_tokens = 8
    attempt_spec = TarsAttemptSpec(
        key=TarsAttemptKey(identity.request_id, identity.attempt_seq),
        request=TarsRequestContext(
            session_id=f"session-{identity.request_id}",
            request_id=identity.request_id,
            input_tokens=input_tokens,
        ),
        evidence=tuple(
            _evidence(target, input_tokens, input_tokens - index - 1) for index, target in enumerate(targets)
        ),
        max_output_tokens=4,
        block_size=4,
    )
    return HandoffAllocateRequest(
        identity=identity,
        attempt_spec=attempt_spec,
        scheduler_epoch=identity.scheduler_epoch,
        expected_topology_epoch=topology_epoch,
        role=role,
    )


def _harness(targets: tuple[HandoffTargetSpec, ...]) -> _Harness:
    runtime = TarsStateRuntime(
        scheduler_id="scheduler-single-role",
        config=TarsOriginalConfig(tpct=1.0, tprt=0.0, dh_first_balance_ttft_threshold=100),
        clock=_fixed_clock,
    )
    runtime.sync_topology(
        1,
        tuple(
            TarsReplicaSpec(
                target=spec.target,
                pending_token_capacity=128,
                running_block_capacity=128,
            )
            for spec in targets
        ),
    )
    server = HandoffProtocolServer(runtime=runtime, scheduler_epoch=7, clock=_fixed_clock)
    assert server.sync_targets(1, targets)
    ledger = HandoffRouterLedger()
    return _Harness(runtime, server, ledger, HandoffProtocolClient(server, ledger))


def _assert_no_transient_state(harness: _Harness) -> None:
    server = harness.server.snapshot()
    runtime = harness.runtime.snapshot()
    assert server.active_attempts == ()
    assert server.pending_deliveries == ()
    assert server.decode_workloads == ()
    assert runtime.attempts == ()
    assert runtime.queue_entries == ()
    assert runtime.reservations == ()
    assert runtime.workloads == ()
    assert harness.ledger.snapshot().entries == ()


def _trigger_harness() -> tuple[_Harness, tuple[TarsTarget, ...], tuple[TarsTarget, ...]]:
    prefill_targets = (_target(100, 0), _target(100, 1))
    decode_targets = (_target(200, 0), _target(200, 1))
    target_specs = tuple(HandoffTargetSpec(target, HandoffRole.PREFILL, "vllm") for target in prefill_targets) + tuple(
        HandoffTargetSpec(target, HandoffRole.DECODE, "vllm") for target in decode_targets
    )
    return _harness(target_specs), prefill_targets, decode_targets


def _hybrid_harness() -> tuple[_Harness, tuple[TarsTarget, ...]]:
    hybrid_targets = (_target(300, 0), _target(300, 1))
    target_specs = tuple(HandoffTargetSpec(target, HandoffRole.UNIFIED, "vllm") for target in hybrid_targets)
    return _harness(target_specs), hybrid_targets


def test_1000_trigger_requests_preserve_d_first_and_single_callback_admission() -> None:
    harness, prefill_targets, decode_targets = _trigger_harness()
    adapter = TriggerTarsAdapter(
        harness.client,
        HandoffAttemptSequencer("trigger-1000", scheduler_epoch=7),
    )
    attempt_sequences: list[int] = []
    dispatched_targets: list[TarsTarget] = []

    for request_number in range(1_000):
        identity = adapter.reserve_attempt()
        attempt_sequences.append(identity.attempt_seq)
        assert harness.runtime.snapshot().attempts == ()
        assert harness.server.snapshot().active_attempts == ()
        assert adapter.bind_decode(
            identity,
            decode_targets[request_number % 2],
            engine_type="vllm",
            topology_epoch=1,
        ).ok
        assert harness.runtime.snapshot().attempts == ()

        def dispatch(
            allocation: HandoffAllocation,
            expected_identity: HandoffAttemptId = identity,
        ) -> None:
            router_entries = harness.ledger.snapshot().entries
            assert allocation.confirmed
            assert len(router_entries) == 1
            assert router_entries[0].identity == expected_identity
            assert router_entries[0].confirmed
            assert router_entries[0].dispatched
            dispatched_targets.append(allocation.target)

        request = _request(identity, prefill_targets, HandoffRole.PREFILL)
        first = adapter.handle_prefill_callback(identity, request, dispatch)
        duplicate = adapter.handle_prefill_callback(
            identity,
            request,
            lambda _allocation: pytest.fail("duplicate callback dispatched Prefill twice"),
        )
        assert first.ok and duplicate.ok
        assert first.allocation == duplicate.allocation
        assert adapter.decode_terminal(identity, aborted=False).ok
        _assert_no_transient_state(harness)

    assert attempt_sequences == list(range(1, 1_001))
    assert adapter.prefill_admission_count == 1_000
    assert adapter.prefill_dispatch_count == 1_000
    assert len(dispatched_targets) == 1_000
    adapter.quiesce()


def test_trigger_rejects_late_mismatch_and_cancelled_callbacks_without_leaks() -> None:
    harness, prefill_targets, decode_targets = _trigger_harness()
    adapter = TriggerTarsAdapter(
        harness.client,
        HandoffAttemptSequencer("trigger-errors", scheduler_epoch=7),
    )

    late_identity = adapter.reserve_attempt()
    assert adapter.bind_decode(late_identity, decode_targets[0], "vllm", 1).ok
    assert adapter.decode_terminal(late_identity, aborted=False).ok
    late = adapter.handle_prefill_callback(
        late_identity,
        _request(late_identity, prefill_targets, HandoffRole.PREFILL),
        lambda _allocation: pytest.fail("late callback dispatched Prefill"),
    )
    assert not late.ok
    assert late.failure is not None
    assert late.failure.code is HandoffFailureCode.STALE_ATTEMPT

    mismatch_identity = adapter.reserve_attempt()
    assert adapter.bind_decode(mismatch_identity, decode_targets[0], "sglang", 1).ok
    mismatch = adapter.handle_prefill_callback(
        mismatch_identity,
        _request(mismatch_identity, prefill_targets, HandoffRole.PREFILL),
        lambda _allocation: pytest.fail("engine mismatch dispatched Prefill"),
    )
    assert not mismatch.ok
    assert mismatch.failure is not None
    assert mismatch.failure.code is HandoffFailureCode.ENGINE_TYPE_MISMATCH
    _assert_no_transient_state(harness)

    topology_identity = adapter.reserve_attempt()
    assert adapter.bind_decode(topology_identity, decode_targets[0], "vllm", 0).ok
    topology = adapter.handle_prefill_callback(
        topology_identity,
        _request(topology_identity, prefill_targets, HandoffRole.PREFILL),
        lambda _allocation: pytest.fail("topology mismatch dispatched Prefill"),
    )
    assert not topology.ok
    assert topology.failure is not None
    assert topology.failure.code is HandoffFailureCode.STALE_TOPOLOGY
    assert adapter.prefill_admission_count == 1
    assert adapter.decode_terminal(topology_identity, aborted=True).ok
    _assert_no_transient_state(harness)

    cancelled_identity = adapter.reserve_attempt()
    assert adapter.bind_decode(cancelled_identity, decode_targets[0], "vllm", 1).ok
    cancellation = asyncio.CancelledError("cancel Trigger callback")

    def cancel_dispatch(_allocation: HandoffAllocation) -> None:
        raise cancellation

    with pytest.raises(asyncio.CancelledError) as caught:
        adapter.handle_prefill_callback(
            cancelled_identity,
            _request(cancelled_identity, prefill_targets, HandoffRole.PREFILL),
            cancel_dispatch,
        )
    assert caught.value is cancellation
    _assert_no_transient_state(harness)


def test_concurrent_trigger_callbacks_share_one_admission_and_dispatch() -> None:
    harness, prefill_targets, decode_targets = _trigger_harness()
    adapter = TriggerTarsAdapter(
        harness.client,
        HandoffAttemptSequencer("trigger-concurrent", scheduler_epoch=7),
    )
    identity = adapter.reserve_attempt()
    assert adapter.bind_decode(identity, decode_targets[0], "vllm", 1).ok
    request = _request(identity, prefill_targets, HandoffRole.PREFILL)
    dispatch_count = 0
    dispatch_lock = Lock()

    def dispatch(_allocation: HandoffAllocation) -> None:
        nonlocal dispatch_count
        with dispatch_lock:
            dispatch_count += 1

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = tuple(
            executor.map(
                lambda _index: adapter.handle_prefill_callback(identity, request, dispatch),
                range(64),
            )
        )

    assert all(result.ok for result in results)
    assert dispatch_count == 1
    assert adapter.prefill_admission_count == 1
    assert adapter.prefill_dispatch_count == 1
    assert adapter.decode_terminal(identity, aborted=False).ok
    _assert_no_transient_state(harness)


def test_1000_pdhybrid_attempts_use_fresh_identity_and_never_emit_prefill_complete() -> None:
    harness, hybrid_targets = _hybrid_harness()
    adapter = PDHybridTarsAdapter(
        harness.client,
        HandoffAttemptSequencer("hybrid-1000", scheduler_epoch=7),
    )
    attempt_sequences: list[int] = []

    for request_number in range(1_000):
        identity = adapter.reserve_attempt()
        attempt_sequences.append(identity.attempt_seq)
        allocation = adapter.allocate(
            identity,
            _request(identity, hybrid_targets, HandoffRole.UNIFIED),
        )
        assert allocation.ok
        assert allocation.allocation is not None
        runtime = harness.runtime.snapshot()
        assert runtime.reservations == ((identity.key, allocation.allocation.target),)
        assert len(runtime.workloads) == 1

        false_prefill = adapter.lifecycle(
            HandoffLifecycleRequest(
                identity,
                HandoffLifecycleEvent.PREFILL_COMPLETE,
                scheduler_epoch=7,
                observed_ttft=0.5,
            )
        )
        assert not false_prefill.ok
        assert false_prefill.failure is not None
        assert false_prefill.failure.code is HandoffFailureCode.INVALID_STATE
        assert harness.runtime.snapshot().reservations == ((identity.key, allocation.allocation.target),)

        terminal_event = HandoffLifecycleEvent.ABORT if request_number < 999 else HandoffLifecycleEvent.REQUEST_COMPLETE
        assert adapter.lifecycle(HandoffLifecycleRequest(identity, terminal_event, scheduler_epoch=7)).ok
        _assert_no_transient_state(harness)

    assert attempt_sequences == list(range(1, 1_001))
    assert adapter.admission_count == 1_000
    assert adapter.physical_prefill_complete_count == 0
    adapter.quiesce()


def test_pdhybrid_late_lifecycle_cannot_mutate_the_new_attempt() -> None:
    harness, hybrid_targets = _hybrid_harness()
    adapter = PDHybridTarsAdapter(
        harness.client,
        HandoffAttemptSequencer("hybrid-late", scheduler_epoch=7),
    )
    old_identity = adapter.reserve_attempt()
    assert adapter.allocate(
        old_identity,
        _request(old_identity, hybrid_targets, HandoffRole.UNIFIED),
    ).ok
    assert adapter.lifecycle(HandoffLifecycleRequest(old_identity, HandoffLifecycleEvent.ABORT, scheduler_epoch=7)).ok

    current_identity = adapter.reserve_attempt()
    current = adapter.allocate(
        current_identity,
        _request(current_identity, hybrid_targets, HandoffRole.UNIFIED),
    )
    assert current.ok
    stale_epoch = adapter.lifecycle(
        HandoffLifecycleRequest(
            current_identity,
            HandoffLifecycleEvent.REQUEST_COMPLETE,
            scheduler_epoch=8,
        )
    )
    assert not stale_epoch.ok
    assert stale_epoch.failure is not None
    assert stale_epoch.failure.code is HandoffFailureCode.STALE_SCHEDULER_EPOCH
    before = harness.runtime.snapshot()
    late = adapter.lifecycle(
        HandoffLifecycleRequest(old_identity, HandoffLifecycleEvent.REQUEST_COMPLETE, scheduler_epoch=7)
    )
    assert not late.ok
    assert late.failure is not None
    assert late.failure.code is HandoffFailureCode.STALE_ATTEMPT
    assert harness.runtime.snapshot() == before

    assert adapter.lifecycle(
        HandoffLifecycleRequest(
            current_identity,
            HandoffLifecycleEvent.REQUEST_COMPLETE,
            scheduler_epoch=7,
        )
    ).ok
    _assert_no_transient_state(harness)
