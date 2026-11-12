# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Atomicity gates for the dormant TARS Bootstrap P/D protocol."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest

from motor.coordinator.scheduler.policy.tars.bootstrap_joint_protocol import (
    BootstrapAttemptId,
    BootstrapAttemptSequencer,
    BootstrapDecodeCandidate,
    BootstrapJointAdmissionSpec,
    BootstrapJointAllocateRequest,
    BootstrapJointBoundaryFailure,
    BootstrapJointConfirmRequest,
    BootstrapJointEvent,
    BootstrapJointFailureCode,
    BootstrapJointFaultPoint,
    BootstrapJointLifecycleRequest,
    BootstrapJointProtocolClient,
    BootstrapJointProtocolServer,
    BootstrapJointRevokeRequest,
    BootstrapJointRole,
    BootstrapJointRouterLedger,
    BootstrapJointState,
    BootstrapJointTargetSpec,
)
from motor.coordinator.scheduler.policy.tars.models import (
    CacheCreditCapability,
    TarsCacheEvidence,
)
from motor.coordinator.scheduler.policy.tars.original.models import (
    TarsOriginalConfig,
    TarsTarget,
)
from motor.coordinator.scheduler.policy.tars.state_runtime import (
    TarsReplicaSpec,
    TarsStateRuntime,
)


@dataclass
class _Harness:
    runtime: TarsStateRuntime
    server: BootstrapJointProtocolServer
    ledger: BootstrapJointRouterLedger
    client: BootstrapJointProtocolClient
    prefill_targets: tuple[TarsTarget, ...]
    decode_targets: tuple[TarsTarget, ...]


class _FaultController:
    def __init__(self) -> None:
        self.point: BootstrapJointFaultPoint | None = None
        self.counts: Counter[BootstrapJointFaultPoint] = Counter()

    def __call__(
        self,
        point: BootstrapJointFaultPoint,
        identity: BootstrapAttemptId,
    ) -> None:
        del identity
        if point is self.point:
            self.counts[point] += 1
            raise BootstrapJointBoundaryFailure(f"injected at {point.value}")


def _target(instance_id: int, dp_rank: int = 0) -> TarsTarget:
    return TarsTarget(instance_id=str(instance_id), dp_rank=dp_rank)


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


def _least_committed_decode(
    admission: BootstrapJointAdmissionSpec,
    candidates: tuple[BootstrapDecodeCandidate, ...],
) -> TarsTarget:
    del admission
    return min(candidates, key=lambda item: (item.committed_tokens, item.target)).target


def _target_specs(
    prefill_targets: tuple[TarsTarget, ...],
    decode_targets: tuple[TarsTarget, ...],
    *,
    circuit_open: TarsTarget | None = None,
    decode_engine_type: str = "sglang",
    decode_capacity: int = 100_000,
) -> tuple[BootstrapJointTargetSpec, ...]:
    return tuple(
        BootstrapJointTargetSpec(
            target,
            BootstrapJointRole.PREFILL,
            "sglang",
            circuit_open=target == circuit_open,
        )
        for target in prefill_targets
    ) + tuple(
        BootstrapJointTargetSpec(
            target,
            BootstrapJointRole.DECODE,
            decode_engine_type,
            circuit_open=target == circuit_open,
            decode_token_capacity=decode_capacity,
        )
        for target in decode_targets
    )


def _harness(
    *,
    fault_controller: _FaultController | None = None,
    decode_selector=_least_committed_decode,
    target_specs: tuple[BootstrapJointTargetSpec, ...] | None = None,
    capacity: int = 100_000,
) -> _Harness:
    prefill_targets = (_target(100, 0), _target(100, 1))
    decode_targets = (_target(200, 0), _target(200, 1))
    all_targets = prefill_targets + decode_targets
    runtime = TarsStateRuntime(
        scheduler_id="scheduler-bootstrap-joint",
        config=TarsOriginalConfig(
            tpct=1.0,
            tprt=0.0,
            dh_first_balance_ttft_threshold=100,
        ),
    )
    runtime.sync_topology(
        1,
        tuple(
            TarsReplicaSpec(
                target=target,
                pending_token_capacity=capacity,
                running_block_capacity=capacity,
            )
            for target in all_targets
        ),
    )
    server = BootstrapJointProtocolServer(
        runtime,
        scheduler_epoch=7,
        decode_selector=decode_selector,
        fault_hook=fault_controller,
    )
    assert server.sync_targets(
        1,
        target_specs or _target_specs(prefill_targets, decode_targets, decode_capacity=capacity),
    )
    ledger = BootstrapJointRouterLedger()
    client = BootstrapJointProtocolClient(server, ledger, fault_hook=fault_controller)
    return _Harness(runtime, server, ledger, client, prefill_targets, decode_targets)


def _admission(
    request_id: str,
    prefill_targets: tuple[TarsTarget, ...],
    *,
    mode: str = "bootstrap",
    scheduler_epoch: int = 7,
    topology_epoch: int = 1,
) -> BootstrapJointAdmissionSpec:
    input_tokens = 8
    return BootstrapJointAdmissionSpec(
        request_id=request_id,
        session_id=f"session-{request_id}",
        input_tokens=input_tokens,
        prefill_evidence=tuple(
            _evidence(target, input_tokens, input_tokens - index - 1) for index, target in enumerate(prefill_targets)
        ),
        max_output_tokens=4,
        block_size=4,
        decode_workload_tokens=12,
        scheduler_epoch=scheduler_epoch,
        expected_topology_epoch=topology_epoch,
        engine_type="sglang",
        coordination_mode=mode,
    )


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
    harness.server.assert_invariants()


def _complete_success(harness: _Harness, request_id: str) -> BootstrapAttemptId:
    admission = _admission(request_id, harness.prefill_targets)
    result = harness.client.allocate(
        admission,
        BootstrapAttemptSequencer(request_id, scheduler_epoch=7),
    )
    assert result.ok and result.allocation is not None
    identity = result.allocation.identity
    dispatched = harness.client.dispatch(identity)
    assert dispatched.ok and dispatched.allocation is not None
    assert dispatched.allocation.dispatch_visible
    prefill = harness.client.lifecycle(
        BootstrapJointLifecycleRequest(
            identity,
            BootstrapJointEvent.PREFILL_COMPLETE,
            scheduler_epoch=7,
            observed_ttft=0.25,
        )
    )
    assert prefill.ok
    completed = harness.client.lifecycle(
        BootstrapJointLifecycleRequest(
            identity,
            BootstrapJointEvent.REQUEST_COMPLETE,
            scheduler_epoch=7,
        )
    )
    assert completed.ok
    return identity


def test_joint_delivery_is_invisible_until_both_workloads_and_confirmation() -> None:
    harness = _harness()
    admission = _admission("atomic", harness.prefill_targets)
    sequencer = BootstrapAttemptSequencer("atomic", scheduler_epoch=7)

    assert harness.server.preflight(admission).ok
    identity = sequencer.reserve()
    allocated = harness.server.allocate(BootstrapJointAllocateRequest(identity, admission))
    assert allocated.ok and allocated.allocation is not None
    assert allocated.attempt is not None
    assert allocated.attempt.state is BootstrapJointState.JOINT_DELIVERY_PENDING
    assert len(harness.runtime.snapshot().workloads) == 1
    assert len(harness.server.snapshot().decode_workloads) == 1
    assert harness.ledger.snapshot().entries == ()

    pending = allocated.allocation
    assert harness.ledger.persist(pending).ok
    before_confirm = harness.ledger.mark_dispatched(identity)
    assert before_confirm.failure is not None
    assert before_confirm.failure.code is BootstrapJointFailureCode.PRECONFIRM_DISPATCH
    assert harness.ledger.snapshot().visible_entries == ()

    confirmed = harness.server.confirm(
        BootstrapJointConfirmRequest(identity, pending.delivery_token, scheduler_epoch=7)
    )
    assert confirmed.ok and confirmed.allocation is not None
    assert harness.ledger.confirm(confirmed.allocation).ok
    visible = harness.ledger.mark_dispatched(identity)
    assert visible.ok and visible.allocation is not None
    assert visible.allocation.prefill_target in harness.prefill_targets
    assert visible.allocation.decode_target in harness.decode_targets
    assert len(harness.ledger.snapshot().visible_entries) == 1

    client = BootstrapJointProtocolClient(harness.server, harness.ledger)
    assert client.lifecycle(
        BootstrapJointLifecycleRequest(
            identity,
            BootstrapJointEvent.PREFILL_COMPLETE,
            scheduler_epoch=7,
        )
    ).ok
    assert harness.runtime.snapshot().workloads == ()
    assert len(harness.server.snapshot().decode_workloads) == 1
    assert client.lifecycle(
        BootstrapJointLifecycleRequest(
            identity,
            BootstrapJointEvent.REQUEST_COMPLETE,
            scheduler_epoch=7,
        )
    ).ok
    _assert_no_transient_state(harness)


@pytest.mark.parametrize("mode", ["handoff", "trigger", "unified"])
def test_non_bootstrap_preflight_has_no_identity_or_state(mode: str) -> None:
    harness = _harness()
    sequencer = BootstrapAttemptSequencer(f"mode-{mode}", scheduler_epoch=7)
    result = harness.client.allocate(
        _admission(f"mode-{mode}", harness.prefill_targets, mode=mode),
        sequencer,
    )
    assert result.failure is not None
    assert result.failure.code is BootstrapJointFailureCode.NON_BOOTSTRAP_MODE
    assert sequencer.reserve().attempt_seq == 1
    _assert_no_transient_state(harness)


def test_preflight_rejects_stale_epoch_without_reserving_attempt() -> None:
    harness = _harness()
    sequencer = BootstrapAttemptSequencer("stale", scheduler_epoch=7)
    result = harness.client.allocate(
        _admission("stale", harness.prefill_targets, topology_epoch=0),
        sequencer,
    )
    assert result.failure is not None
    assert result.failure.code is BootstrapJointFailureCode.STALE_TOPOLOGY
    assert sequencer.reserve().attempt_seq == 1
    _assert_no_transient_state(harness)


@pytest.mark.parametrize(
    ("selector", "circuit_target", "expected_code"),
    [
        (
            lambda admission, candidates: admission.prefill_evidence[0].target,
            None,
            BootstrapJointFailureCode.TARGET_MISMATCH,
        ),
        (
            lambda _admission, _candidates: _target(200, 0),
            _target(200, 0),
            BootstrapJointFailureCode.CIRCUIT_OPEN,
        ),
    ],
)
def test_final_pair_validation_compensates_before_visibility(
    selector,
    circuit_target: TarsTarget | None,
    expected_code: BootstrapJointFailureCode,
) -> None:
    prefill_targets = (_target(100, 0), _target(100, 1))
    decode_targets = (_target(200, 0), _target(200, 1))
    harness = _harness(
        decode_selector=selector,
        target_specs=_target_specs(
            prefill_targets,
            decode_targets,
            circuit_open=circuit_target,
        ),
    )
    result = harness.client.allocate(
        _admission("invalid-pair", harness.prefill_targets),
        BootstrapAttemptSequencer("invalid-pair", scheduler_epoch=7),
    )
    assert result.failure is not None
    assert result.failure.code is expected_code
    _assert_no_transient_state(harness)


def test_duplicate_joint_admission_is_rejected_without_join_or_takeover() -> None:
    harness = _harness()
    admission = _admission("duplicate", harness.prefill_targets)
    identity = BootstrapAttemptSequencer("duplicate", scheduler_epoch=7).reserve()
    request = BootstrapJointAllocateRequest(identity, admission)
    first = harness.server.allocate(request)
    second = harness.server.allocate(request)
    assert first.ok and first.allocation is not None
    assert second.failure is not None
    assert second.failure.code is BootstrapJointFailureCode.ATTEMPT_CONFLICT
    assert harness.server.revoke(
        BootstrapJointRevokeRequest(
            identity,
            first.allocation.delivery_token,
            scheduler_epoch=7,
        )
    ).ok
    _assert_no_transient_state(harness)


def test_every_joint_boundary_fault_compensates_100_percent() -> None:
    controller = _FaultController()
    harness = _harness(fault_controller=controller)

    for point in BootstrapJointFaultPoint:
        controller.point = point
        for iteration in range(100):
            request_id = f"fault-{point.value}-{iteration}"
            result = harness.client.allocate(
                _admission(request_id, harness.prefill_targets),
                BootstrapAttemptSequencer(request_id, scheduler_epoch=7),
            )
            if point in (
                BootstrapJointFaultPoint.AFTER_PREFILL_RELEASE,
                BootstrapJointFaultPoint.AFTER_DECODE_RELEASE,
            ):
                assert result.ok and result.allocation is not None
                identity = result.allocation.identity
                assert harness.client.dispatch(identity).ok
                result = harness.client.lifecycle(
                    BootstrapJointLifecycleRequest(
                        identity,
                        BootstrapJointEvent.PREFILL_COMPLETE,
                        scheduler_epoch=7,
                    )
                )
                if point is BootstrapJointFaultPoint.AFTER_DECODE_RELEASE:
                    assert result.ok
                    result = harness.client.lifecycle(
                        BootstrapJointLifecycleRequest(
                            identity,
                            BootstrapJointEvent.REQUEST_COMPLETE,
                            scheduler_epoch=7,
                        )
                    )
            assert result.failure is not None
            assert result.failure.code is BootstrapJointFailureCode.BOUNDARY_FAILURE
            _assert_no_transient_state(harness)
        assert controller.counts[point] == 100


def test_10000_state_simulations_preserve_joint_invariants() -> None:
    completed = 0
    rejected_non_bootstrap = 0
    aborted = 0

    for batch in range(100):
        harness = _harness()
        for offset in range(100):
            number = batch * 100 + offset
            request_id = f"simulation-{number}"
            if number % 5 == 0:
                sequencer = BootstrapAttemptSequencer(request_id, scheduler_epoch=7)
                result = harness.client.allocate(
                    _admission(request_id, harness.prefill_targets, mode="handoff"),
                    sequencer,
                )
                assert result.failure is not None
                assert result.failure.code is BootstrapJointFailureCode.NON_BOOTSTRAP_MODE
                assert sequencer.reserve().attempt_seq == 1
                rejected_non_bootstrap += 1
            else:
                admission = _admission(request_id, harness.prefill_targets)
                result = harness.client.allocate(
                    admission,
                    BootstrapAttemptSequencer(request_id, scheduler_epoch=7),
                )
                assert result.ok and result.allocation is not None
                identity = result.allocation.identity
                dispatched = harness.client.dispatch(identity)
                assert dispatched.ok and dispatched.allocation is not None
                assert dispatched.allocation.prefill_target in harness.prefill_targets
                assert dispatched.allocation.decode_target in harness.decode_targets
                if number % 5 == 1:
                    assert harness.client.lifecycle(
                        BootstrapJointLifecycleRequest(
                            identity,
                            BootstrapJointEvent.ABORT,
                            scheduler_epoch=7,
                        )
                    ).ok
                    aborted += 1
                else:
                    assert harness.client.lifecycle(
                        BootstrapJointLifecycleRequest(
                            identity,
                            BootstrapJointEvent.PREFILL_COMPLETE,
                            scheduler_epoch=7,
                        )
                    ).ok
                    assert harness.client.lifecycle(
                        BootstrapJointLifecycleRequest(
                            identity,
                            BootstrapJointEvent.PREFILL_COMPLETE,
                            scheduler_epoch=7,
                        )
                    ).ok
                    assert harness.client.lifecycle(
                        BootstrapJointLifecycleRequest(
                            identity,
                            BootstrapJointEvent.REQUEST_COMPLETE,
                            scheduler_epoch=7,
                        )
                    ).ok
                    completed += 1
            _assert_no_transient_state(harness)

    assert completed == 6_000
    assert rejected_non_bootstrap == 2_000
    assert aborted == 2_000


def test_1000_concurrent_joint_requests_converge_all_transient_ledgers() -> None:
    harness = _harness(capacity=1_000_000)

    def run_request(number: int) -> BootstrapAttemptId:
        request_id = f"concurrent-{number}"
        identity = _complete_success(harness, request_id)
        return identity

    with ThreadPoolExecutor(max_workers=32) as pool:
        identities = tuple(pool.map(run_request, range(1_000)))

    assert len(set(identities)) == 1_000
    assert all(identity.attempt_seq == 1 for identity in identities)
    _assert_no_transient_state(harness)
