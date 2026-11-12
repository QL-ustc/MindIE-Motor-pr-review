# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""State invariants for the dormant Scheduler-owned TARS runtime."""

import random
from unittest.mock import Mock

import pytest

from motor.config.coordinator import SchedulerType
from motor.coordinator.scheduler.policy.tars.models import (
    CacheCreditCapability,
    TarsCacheEvidence,
    TarsTarget,
)
from motor.coordinator.scheduler.policy.tars.original.models import (
    TarsOriginalConfig,
    TarsRequestContext,
)
from motor.coordinator.scheduler.policy.tars.state_runtime import (
    TarsAttemptKey,
    TarsAttemptPhase,
    TarsAttemptSpec,
    TarsCapacityError,
    TarsFinalTargetLockedError,
    TarsReplicaSpec,
    TarsStaleTopologyError,
    TarsStateRuntime,
)
from motor.coordinator.scheduler.scheduler import Scheduler


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _target(instance: int, dp_rank: int = 0) -> TarsTarget:
    return TarsTarget(instance_id=str(instance), dp_rank=dp_rank)


def _replica(
    target: TarsTarget,
    *,
    pending_capacity: int = 64,
    running_blocks: int = 64,
) -> TarsReplicaSpec:
    return TarsReplicaSpec(
        target=target,
        pending_token_capacity=pending_capacity,
        running_block_capacity=running_blocks,
    )


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


def _attempt(
    request_number: int,
    targets: tuple[TarsTarget, ...],
    *,
    input_tokens: int = 10,
    credit_tokens_by_target: tuple[int, ...] | None = None,
    max_output_tokens: int = 0,
    block_size: int = 4,
) -> TarsAttemptSpec:
    if credit_tokens_by_target is None:
        credit_tokens_by_target = tuple(0 for _ in targets)
    key = TarsAttemptKey(request_id=f"request-{request_number}", attempt_seq=request_number % 3)
    return TarsAttemptSpec(
        key=key,
        request=TarsRequestContext(
            session_id=f"session-{request_number}",
            request_id=key.request_id,
            input_tokens=input_tokens,
        ),
        evidence=tuple(
            _evidence(target, input_tokens, credit)
            for target, credit in zip(targets, credit_tokens_by_target, strict=True)
        ),
        max_output_tokens=max_output_tokens,
        block_size=block_size,
    )


def _attempts_by_key(runtime: TarsStateRuntime):
    return {attempt.key: attempt for attempt in runtime.snapshot().attempts}


def _replicas_by_target(runtime: TarsStateRuntime):
    return {replica.target: replica for replica in runtime.snapshot().replicas}


def test_scheduler_lazily_owns_exactly_one_tars_runtime():
    scheduler = Scheduler(instance_provider=Mock(), config=SchedulerType.LOAD_BALANCE)
    clock = FakeClock()

    assert scheduler.get_tars_state_runtime() is None

    first = scheduler.create_tars_state_runtime(
        scheduler_id="scheduler-process-7",
        config=TarsOriginalConfig(tpct=1.0, tprt=0.0),
        clock=clock,
    )
    second = scheduler.create_tars_state_runtime(
        scheduler_id="scheduler-process-7",
        config=TarsOriginalConfig(tpct=1.0, tprt=0.0),
        clock=clock,
    )

    assert first is second
    assert scheduler.get_tars_state_runtime() is first
    with pytest.raises(RuntimeError, match="already owns"):
        scheduler.create_tars_state_runtime(scheduler_id="another-scheduler", clock=clock)


def test_runtime_tracks_pending_ttft_budget_and_releases_all_transient_state():
    clock = FakeClock()
    first = _target(10, 0)
    second = _target(11, 0)
    runtime = TarsStateRuntime(
        scheduler_id="scheduler-1",
        config=TarsOriginalConfig(tpct=1.0, tprt=0.0, dh_first_balance_ttft_threshold=100),
        clock=clock,
    )
    assert runtime.sync_topology(1, (_replica(first, pending_capacity=10), _replica(second)))
    spec = _attempt(1, (first, second), credit_tokens_by_target=(8, 2), max_output_tokens=6)

    queued = runtime.enqueue(spec)
    assert queued.phase is TarsAttemptPhase.QUEUED
    assert queued.selected_target == first
    assert queued.final_target is None

    reserved = runtime.reserve(spec.key, expected_topology_epoch=1)
    assert reserved.phase is TarsAttemptPhase.RESERVED
    assert reserved.owner_target == first
    assert reserved.final_target == first
    snapshot = runtime.snapshot()
    replica = _replicas_by_target(runtime)[first]
    assert snapshot.reservations == ((spec.key, first),)
    assert len(snapshot.workloads) == 1
    assert replica.pending_compute_tokens == 2
    assert replica.pending_budget_remaining == 8
    assert replica.reserved_blocks == 4
    assert replica.pending_ttft_remaining == 2.0

    clock.advance(0.75)
    assert _replicas_by_target(runtime)[first].pending_ttft_remaining == 1.25
    assert runtime.complete_prefill(spec.key, observed_ttft=1.2, keep_running=True)
    running = _attempts_by_key(runtime)[spec.key]
    replica = _replicas_by_target(runtime)[first]
    assert running.phase is TarsAttemptPhase.RUNNING
    assert replica.pending_compute_tokens == 0
    assert replica.pending_budget_remaining == 10
    assert replica.reserved_blocks == 4
    assert replica.last_ttft == 1.2

    assert runtime.complete_request(spec.key)
    snapshot = runtime.snapshot()
    assert snapshot.attempts == ()
    assert snapshot.queue_entries == ()
    assert snapshot.reservations == ()
    assert snapshot.workloads == ()
    assert {replica.target for replica in snapshot.replicas} == {first, second}


def test_capacity_rebalance_moves_once_and_prevents_a_cycle():
    clock = FakeClock()
    first = _target(20, 0)
    second = _target(21, 0)
    runtime = TarsStateRuntime(
        scheduler_id="scheduler-2",
        config=TarsOriginalConfig(tpct=1.0, tprt=0.0, dh_first_balance_ttft_threshold=100),
        clock=clock,
    )
    runtime.sync_topology(
        1,
        (
            _replica(first, pending_capacity=5, running_blocks=3),
            _replica(second, pending_capacity=10, running_blocks=3),
        ),
    )
    blocker = _attempt(2, (first, second), input_tokens=10, credit_tokens_by_target=(5, 0))
    waiting = _attempt(3, (first, second), input_tokens=4, credit_tokens_by_target=(4, 0))

    assert runtime.enqueue(blocker).selected_target == first
    runtime.reserve(blocker.key, expected_topology_epoch=1)
    assert runtime.enqueue(waiting).selected_target == first

    with pytest.raises(TarsCapacityError):
        runtime.lock_final_target(waiting.key, expected_topology_epoch=1)
    assert _attempts_by_key(runtime)[waiting.key].final_target is None

    assert runtime.rebalance(waiting.key, expected_topology_epoch=1)
    assert _attempts_by_key(runtime)[waiting.key].selected_target == second

    assert runtime.complete_prefill(blocker.key, observed_ttft=1.0, keep_running=False)
    assert not runtime.rebalance(waiting.key, expected_topology_epoch=1)
    snapshot = runtime.snapshot()
    assert _attempts_by_key(runtime)[waiting.key].selected_target == second
    assert snapshot.rebalance_count == 1
    assert snapshot.cycle_prevention_count == 1
    runtime.assert_invariants()


def test_final_target_survives_new_topology_and_stale_epoch_cannot_mutate_it():
    clock = FakeClock()
    first = _target(30, 0)
    second = _target(31, 0)
    runtime = TarsStateRuntime(scheduler_id="scheduler-3", clock=clock)
    runtime.sync_topology(1, (_replica(first), _replica(second)))
    spec = _attempt(4, (first, second), credit_tokens_by_target=(9, 0))

    queued = runtime.enqueue(spec)
    assert queued.selected_target == first
    locked = runtime.lock_final_target(spec.key, expected_topology_epoch=1)
    assert locked.final_target == first

    assert runtime.sync_topology(2, (_replica(second),))
    assert not runtime.sync_topology(2, (_replica(first), _replica(second)))
    assert not runtime.sync_topology(1, (_replica(first), _replica(second)))
    current = _attempts_by_key(runtime)[spec.key]
    assert runtime.topology_epoch == 2
    assert current.selected_target == first
    assert current.final_target == first
    assert not _replicas_by_target(runtime)[first].active

    with pytest.raises(TarsStaleTopologyError):
        runtime.rebalance(spec.key, expected_topology_epoch=1)
    with pytest.raises(TarsFinalTargetLockedError):
        runtime.rebalance(spec.key, expected_topology_epoch=2)
    with pytest.raises(TarsCapacityError):
        runtime.reserve(spec.key, expected_topology_epoch=2)
    assert _attempts_by_key(runtime)[spec.key].final_target == first

    assert runtime.abort(spec.key)
    assert set(_replicas_by_target(runtime)) == {second}


def test_10000_randomized_transitions_preserve_runtime_invariants_and_quiesce():
    rng = random.Random(20260828)
    clock = FakeClock()
    targets = tuple(_target(40 + index // 2, index % 2) for index in range(4))
    runtime = TarsStateRuntime(
        scheduler_id="scheduler-random",
        config=TarsOriginalConfig(tpct=0.5, tprt=0.0, dh_first_balance_ttft_threshold=16),
        clock=clock,
    )
    epoch = 1
    active_targets = set(targets)
    runtime.sync_topology(epoch, tuple(_replica(target, pending_capacity=96) for target in targets))
    next_request = 100
    locked_targets: dict[TarsAttemptKey, TarsTarget] = {}

    for transition in range(10_000):
        snapshot = runtime.snapshot()
        attempts = {attempt.key: attempt for attempt in snapshot.attempts}
        action = rng.randrange(7)
        try:
            if action == 0 and len(attempts) < 24:
                input_tokens = rng.randrange(1, 33)
                credit_tokens_by_target = tuple(rng.randrange(0, input_tokens + 1) for _ in targets)
                runtime.enqueue(
                    _attempt(
                        next_request,
                        targets,
                        input_tokens=input_tokens,
                        credit_tokens_by_target=credit_tokens_by_target,
                        max_output_tokens=rng.randrange(0, 17),
                        block_size=rng.choice((4, 8, 16)),
                    )
                )
                next_request += 1
            elif action == 1:
                queued = [attempt for attempt in attempts.values() if attempt.phase is TarsAttemptPhase.QUEUED]
                if queued:
                    runtime.rebalance(rng.choice(queued).key, expected_topology_epoch=epoch)
            elif action == 2:
                queued = [attempt for attempt in attempts.values() if attempt.phase is TarsAttemptPhase.QUEUED]
                if queued:
                    runtime.reserve(rng.choice(queued).key, expected_topology_epoch=epoch)
            elif action == 3:
                reserved = [attempt for attempt in attempts.values() if attempt.phase is TarsAttemptPhase.RESERVED]
                if reserved:
                    runtime.complete_prefill(
                        rng.choice(reserved).key,
                        observed_ttft=round(rng.random() * 4, 4),
                        keep_running=bool(rng.getrandbits(1)),
                    )
            elif action == 4:
                terminal = [
                    attempt
                    for attempt in attempts.values()
                    if attempt.phase in (TarsAttemptPhase.RESERVED, TarsAttemptPhase.RUNNING)
                ]
                if terminal:
                    runtime.complete_request(rng.choice(terminal).key)
            elif action == 5 and attempts:
                runtime.abort(rng.choice(tuple(attempts)))
            elif action == 6 and transition % 17 == 0:
                epoch += 1
                active_targets = set(rng.sample(list(targets), rng.randrange(2, len(targets) + 1)))
                runtime.sync_topology(
                    epoch,
                    tuple(
                        _replica(
                            target,
                            pending_capacity=rng.randrange(48, 129),
                            running_blocks=rng.randrange(24, 65),
                        )
                        for target in sorted(active_targets)
                    ),
                )
        except (TarsCapacityError, TarsFinalTargetLockedError):
            pass

        clock.advance(rng.random() / 100)
        runtime.assert_invariants()
        current = runtime.snapshot()
        queue_counts: dict[TarsAttemptKey, int] = {}
        for _, key in current.queue_entries:
            queue_counts[key] = queue_counts.get(key, 0) + 1
        assert all(count == 1 for count in queue_counts.values())
        assert len({key for key, _ in current.reservations}) == len(current.reservations)
        for attempt in current.attempts:
            if attempt.final_target is not None:
                previous = locked_targets.setdefault(attempt.key, attempt.final_target)
                assert attempt.final_target == previous

    runtime.quiesce()
    final = runtime.snapshot()
    runtime.assert_invariants()
    assert final.attempts == ()
    assert final.queue_entries == ()
    assert final.reservations == ()
    assert final.workloads == ()
    assert {replica.target for replica in final.replicas} == active_targets
