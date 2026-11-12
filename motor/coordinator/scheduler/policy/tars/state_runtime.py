# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Dormant Scheduler-owned TARS queue and replica state.

This module owns no transport or Motor workload side effects. It only models
the state that later protocol layers will drive explicitly.
"""

import math
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum

from motor.coordinator.scheduler.policy.tars.models import TarsCacheEvidence
from motor.coordinator.scheduler.policy.tars.original.models import (
    TarsCandidate,
    TarsOriginalConfig,
    TarsRequestContext,
    TarsTarget,
)
from motor.coordinator.scheduler.policy.tars.original.scorer import estimate_ttft
from motor.coordinator.scheduler.policy.tars.original.selector import select_tars_candidate


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class TarsStateError(RuntimeError):
    """Base error for rejected dormant-state transitions."""


class TarsStaleTopologyError(TarsStateError):
    """A transition was planned against an obsolete topology epoch."""


class TarsFinalTargetLockedError(TarsStateError):
    """A caller tried to retarget an attempt after its final target was locked."""


class TarsCapacityError(TarsStateError):
    """No active replica can reserve the attempt's token and block demand."""


@dataclass(frozen=True, order=True)
class TarsAttemptKey:
    request_id: str
    attempt_seq: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not _is_nonnegative_int(self.attempt_seq):
            raise ValueError("attempt_seq must be a non-negative integer")


@dataclass(frozen=True)
class TarsReplicaSpec:
    target: TarsTarget
    pending_token_capacity: int
    running_block_capacity: int

    def __post_init__(self) -> None:
        if not _is_nonnegative_int(self.pending_token_capacity):
            raise ValueError("pending_token_capacity must be a non-negative integer")
        if not _is_nonnegative_int(self.running_block_capacity):
            raise ValueError("running_block_capacity must be a non-negative integer")


@dataclass(frozen=True)
class TarsAttemptSpec:
    key: TarsAttemptKey
    request: TarsRequestContext
    evidence: tuple[TarsCacheEvidence, ...]
    max_output_tokens: int = 0
    block_size: int = 128

    def __post_init__(self) -> None:
        if self.request.request_id != self.key.request_id:
            raise ValueError("attempt and request IDs must agree")
        if not self.evidence:
            raise ValueError("attempt evidence must not be empty")
        if not _is_nonnegative_int(self.max_output_tokens):
            raise ValueError("max_output_tokens must be a non-negative integer")
        if not _is_nonnegative_int(self.block_size) or self.block_size == 0:
            raise ValueError("block_size must be a positive integer")
        targets = [item.target for item in self.evidence]
        if len(targets) != len(set(targets)):
            raise ValueError("attempt evidence targets must be unique")
        if any(item.input_tokens != self.request.input_tokens for item in self.evidence):
            raise ValueError("attempt evidence input tokens must match the request")

    @property
    def evidence_by_target(self) -> dict[TarsTarget, TarsCacheEvidence]:
        return {item.target: item for item in self.evidence}

    @property
    def reserved_blocks(self) -> int:
        return math.ceil((self.request.input_tokens + self.max_output_tokens) / self.block_size)


class TarsAttemptPhase(str, Enum):
    QUEUED = "queued"
    RESERVED = "reserved"
    RUNNING = "running"


@dataclass(frozen=True)
class TarsAttemptSnapshot:
    key: TarsAttemptKey
    phase: TarsAttemptPhase
    selected_target: TarsTarget
    final_target: TarsTarget | None
    owner_target: TarsTarget | None
    topology_epoch: int
    queued_at: float
    visited_targets: tuple[TarsTarget, ...]


@dataclass(frozen=True)
class TarsWorkloadSnapshot:
    key: TarsAttemptKey
    target: TarsTarget
    phase: TarsAttemptPhase
    compute_tokens: int
    cache_credit_tokens: int
    reserved_blocks: int
    reserved_at: float
    estimated_ttft: float


@dataclass(frozen=True)
class TarsReplicaSnapshot:
    target: TarsTarget
    active: bool
    pending_token_capacity: int
    pending_compute_tokens: int
    pending_budget_remaining: int
    running_block_capacity: int
    reserved_blocks: int
    pending_ttft_remaining: float
    last_ttft: float


@dataclass(frozen=True)
class TarsRuntimeSnapshot:
    scheduler_id: str
    topology_epoch: int
    attempts: tuple[TarsAttemptSnapshot, ...]
    replicas: tuple[TarsReplicaSnapshot, ...]
    queue_entries: tuple[tuple[TarsTarget, TarsAttemptKey], ...]
    reservations: tuple[tuple[TarsAttemptKey, TarsTarget], ...]
    workloads: tuple[TarsWorkloadSnapshot, ...]
    rebalance_count: int
    cycle_prevention_count: int


@dataclass
class _AttemptState:
    spec: TarsAttemptSpec
    phase: TarsAttemptPhase
    selected_target: TarsTarget
    topology_epoch: int
    queued_at: float
    visited_targets: list[TarsTarget]
    final_target: TarsTarget | None = None
    owner_target: TarsTarget | None = None


@dataclass
class _WorkloadState:
    key: TarsAttemptKey
    target: TarsTarget
    phase: TarsAttemptPhase
    compute_tokens: int
    cache_credit_tokens: int
    reserved_blocks: int
    reserved_at: float
    estimated_ttft: float


@dataclass
class _ReplicaState:
    spec: TarsReplicaSpec
    active: bool = True
    pending: set[TarsAttemptKey] = field(default_factory=set)
    running: set[TarsAttemptKey] = field(default_factory=set)
    last_ttft: float = 0.0


class TarsStateRuntime:
    """Single aggregate owning dormant TARS queue, reservation, and replica state."""

    def __init__(
        self,
        scheduler_id: str,
        config: TarsOriginalConfig | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(scheduler_id, str) or not scheduler_id:
            raise ValueError("scheduler_id must be a non-empty string")
        self.scheduler_id = scheduler_id
        self._config = config or TarsOriginalConfig()
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._topology_epoch = -1
        self._replicas: dict[TarsTarget, _ReplicaState] = {}
        self._attempts: dict[TarsAttemptKey, _AttemptState] = {}
        self._queues: dict[TarsTarget, list[TarsAttemptKey]] = {}
        self._reservations: dict[TarsAttemptKey, TarsTarget] = {}
        self._workloads: dict[TarsAttemptKey, _WorkloadState] = {}
        self._rebalance_count = 0
        self._cycle_prevention_count = 0

    @property
    def topology_epoch(self) -> int:
        return self._topology_epoch

    def sync_topology(self, epoch: int, replicas: Iterable[TarsReplicaSpec]) -> bool:
        """Install a strictly newer topology without retargeting locked attempts."""
        if not _is_nonnegative_int(epoch):
            raise ValueError("topology epoch must be a non-negative integer")
        replica_specs = tuple(replicas)
        targets = [replica.target for replica in replica_specs]
        if len(targets) != len(set(targets)):
            raise ValueError("topology replica targets must be unique")

        with self._lock:
            if epoch <= self._topology_epoch:
                return False
            for replica in self._replicas.values():
                replica.active = False
            for spec in replica_specs:
                replica = self._replicas.get(spec.target)
                if replica is None:
                    self._replicas[spec.target] = _ReplicaState(spec=spec)
                else:
                    replica.spec = spec
                    replica.active = True
                self._queues.setdefault(spec.target, [])
            self._topology_epoch = epoch
            self._prune_inactive_replicas_locked()
            self._assert_invariants_locked()
            return True

    def enqueue(self, spec: TarsAttemptSpec) -> TarsAttemptSnapshot:
        with self._lock:
            if spec.key in self._attempts:
                raise TarsStateError("attempt already exists")
            selected = self._select_target_locked(spec, require_capacity=False, exclude_key=None)
            now = float(self._clock())
            attempt = _AttemptState(
                spec=spec,
                phase=TarsAttemptPhase.QUEUED,
                selected_target=selected,
                topology_epoch=self._topology_epoch,
                queued_at=now,
                visited_targets=[selected],
            )
            self._attempts[spec.key] = attempt
            self._queues.setdefault(selected, []).append(spec.key)
            self._assert_invariants_locked()
            return self._snapshot_attempt_locked(attempt)

    def rebalance(self, key: TarsAttemptKey, *, expected_topology_epoch: int) -> bool:
        with self._lock:
            self._require_current_epoch_locked(expected_topology_epoch)
            attempt = self._get_attempt_locked(key)
            if attempt.phase is not TarsAttemptPhase.QUEUED:
                raise TarsStateError("only queued attempts can be rebalanced")
            if attempt.final_target is not None:
                raise TarsFinalTargetLockedError("final target is locked")
            selected = self._select_target_locked(
                attempt.spec,
                require_capacity=True,
                exclude_key=key,
            )
            if selected == attempt.selected_target:
                return False
            if selected in attempt.visited_targets:
                self._cycle_prevention_count += 1
                return False
            self._remove_queue_entry_locked(attempt.selected_target, key)
            self._queues.setdefault(selected, []).append(key)
            attempt.selected_target = selected
            attempt.topology_epoch = self._topology_epoch
            attempt.visited_targets.append(selected)
            self._rebalance_count += 1
            self._assert_invariants_locked()
            return True

    def lock_final_target(
        self,
        key: TarsAttemptKey,
        *,
        expected_topology_epoch: int,
    ) -> TarsAttemptSnapshot:
        with self._lock:
            self._require_current_epoch_locked(expected_topology_epoch)
            attempt = self._get_attempt_locked(key)
            if attempt.phase is not TarsAttemptPhase.QUEUED:
                raise TarsStateError("only queued attempts can lock a final target")
            if attempt.final_target is None:
                evidence = attempt.spec.evidence_by_target[attempt.selected_target]
                if not self._can_reserve_locked(
                    attempt.selected_target,
                    evidence,
                    attempt.spec.reserved_blocks,
                ):
                    raise TarsCapacityError("selected target has insufficient capacity")
                attempt.final_target = attempt.selected_target
            self._assert_invariants_locked()
            return self._snapshot_attempt_locked(attempt)

    def release_final_target(
        self,
        key: TarsAttemptKey,
        *,
        expected_topology_epoch: int,
    ) -> TarsAttemptSnapshot:
        """Release a pre-workload target lock while keeping the attempt queued."""
        with self._lock:
            self._require_current_epoch_locked(expected_topology_epoch)
            attempt = self._get_attempt_locked(key)
            if attempt.phase is not TarsAttemptPhase.QUEUED:
                raise TarsStateError("only queued attempts can release a final target")
            if attempt.final_target is None:
                raise TarsStateError("attempt has no final target to release")
            attempt.final_target = None
            self._assert_invariants_locked()
            return self._snapshot_attempt_locked(attempt)

    def reserve(
        self,
        key: TarsAttemptKey,
        *,
        expected_topology_epoch: int,
    ) -> TarsAttemptSnapshot:
        with self._lock:
            self._require_current_epoch_locked(expected_topology_epoch)
            attempt = self._get_attempt_locked(key)
            if attempt.phase is not TarsAttemptPhase.QUEUED:
                raise TarsStateError("only queued attempts can reserve capacity")
            target = attempt.final_target or attempt.selected_target
            evidence = attempt.spec.evidence_by_target[target]
            if not self._can_reserve_locked(target, evidence, attempt.spec.reserved_blocks):
                raise TarsCapacityError("selected target has insufficient capacity")

            self._remove_queue_entry_locked(attempt.selected_target, key)
            attempt.final_target = target
            attempt.owner_target = target
            attempt.phase = TarsAttemptPhase.RESERVED
            attempt.topology_epoch = self._topology_epoch
            estimated_ttft = estimate_ttft(
                evidence.estimated_compute_tokens,
                evidence.best_effort_cache_credit_tokens,
                config=self._config,
            )
            workload = _WorkloadState(
                key=key,
                target=target,
                phase=TarsAttemptPhase.RESERVED,
                compute_tokens=evidence.estimated_compute_tokens,
                cache_credit_tokens=evidence.best_effort_cache_credit_tokens,
                reserved_blocks=attempt.spec.reserved_blocks,
                reserved_at=float(self._clock()),
                estimated_ttft=estimated_ttft,
            )
            self._reservations[key] = target
            self._workloads[key] = workload
            self._replicas[target].pending.add(key)
            self._assert_invariants_locked()
            return self._snapshot_attempt_locked(attempt)

    def complete_prefill(
        self,
        key: TarsAttemptKey,
        observed_ttft: float | None,
        *,
        keep_running: bool,
    ) -> bool:
        with self._lock:
            attempt = self._attempts.get(key)
            if attempt is None or attempt.phase is not TarsAttemptPhase.RESERVED:
                return False
            workload = self._workloads[key]
            replica = self._replicas[workload.target]
            replica.pending.discard(key)
            valid_ttft = self._valid_ttft(observed_ttft)
            if valid_ttft is not None:
                replica.last_ttft = valid_ttft
            if keep_running:
                attempt.phase = TarsAttemptPhase.RUNNING
                workload.phase = TarsAttemptPhase.RUNNING
                replica.running.add(key)
            else:
                self._finish_attempt_locked(key)
            self._prune_inactive_replicas_locked()
            self._assert_invariants_locked()
            return True

    def complete_request(self, key: TarsAttemptKey) -> bool:
        with self._lock:
            if key not in self._attempts:
                return False
            self._finish_attempt_locked(key)
            self._prune_inactive_replicas_locked()
            self._assert_invariants_locked()
            return True

    def abort(self, key: TarsAttemptKey) -> bool:
        with self._lock:
            if key not in self._attempts:
                return False
            self._finish_attempt_locked(key)
            self._prune_inactive_replicas_locked()
            self._assert_invariants_locked()
            return True

    def quiesce(self) -> None:
        """Release every transient attempt while retaining active topology inventory."""
        with self._lock:
            for key in tuple(self._attempts):
                self._finish_attempt_locked(key)
            self._prune_inactive_replicas_locked()
            self._assert_invariants_locked()

    def snapshot(self) -> TarsRuntimeSnapshot:
        with self._lock:
            attempts = tuple(self._snapshot_attempt_locked(attempt) for _, attempt in sorted(self._attempts.items()))
            replicas = tuple(
                self._snapshot_replica_locked(target, replica) for target, replica in sorted(self._replicas.items())
            )
            queue_entries = tuple((target, key) for target, keys in sorted(self._queues.items()) for key in keys)
            reservations = tuple(sorted(self._reservations.items()))
            workloads = tuple(self._snapshot_workload(workload) for _, workload in sorted(self._workloads.items()))
            return TarsRuntimeSnapshot(
                scheduler_id=self.scheduler_id,
                topology_epoch=self._topology_epoch,
                attempts=attempts,
                replicas=replicas,
                queue_entries=queue_entries,
                reservations=reservations,
                workloads=workloads,
                rebalance_count=self._rebalance_count,
                cycle_prevention_count=self._cycle_prevention_count,
            )

    def assert_invariants(self) -> None:
        with self._lock:
            self._assert_invariants_locked()

    @staticmethod
    def _valid_ttft(value: float | None) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result) or result < 0:
            return None
        return round(result, 4)

    def _require_current_epoch_locked(self, expected_epoch: int) -> None:
        if expected_epoch != self._topology_epoch:
            raise TarsStaleTopologyError(
                f"stale topology epoch: expected {expected_epoch}, current {self._topology_epoch}"
            )

    def _get_attempt_locked(self, key: TarsAttemptKey) -> _AttemptState:
        attempt = self._attempts.get(key)
        if attempt is None:
            raise TarsStateError("unknown attempt")
        return attempt

    def _select_target_locked(
        self,
        spec: TarsAttemptSpec,
        *,
        require_capacity: bool,
        exclude_key: TarsAttemptKey | None,
    ) -> TarsTarget:
        candidates = []
        for evidence in spec.evidence:
            replica = self._replicas.get(evidence.target)
            if replica is None or not replica.active:
                continue
            if require_capacity and not self._can_reserve_locked(
                evidence.target,
                evidence,
                spec.reserved_blocks,
            ):
                continue
            waiting_compute = 0
            waiting_credit = 0
            for queued_key in self._queues.get(evidence.target, ()):
                if queued_key == exclude_key:
                    continue
                queued_evidence = self._attempts[queued_key].spec.evidence_by_target.get(evidence.target)
                if queued_evidence is not None:
                    waiting_compute += queued_evidence.estimated_compute_tokens
                    waiting_credit += queued_evidence.best_effort_cache_credit_tokens
            pending_compute = 0
            pending_credit = 0
            running_blocks = 0
            for workload in self._workloads.values():
                if workload.target != evidence.target:
                    continue
                if workload.phase is TarsAttemptPhase.RESERVED:
                    pending_compute += workload.compute_tokens
                    pending_credit += workload.cache_credit_tokens
                elif workload.phase is TarsAttemptPhase.RUNNING:
                    running_blocks += workload.reserved_blocks
            candidates.append(
                TarsCandidate(
                    target=evidence.target,
                    estimated_compute_tokens=evidence.estimated_compute_tokens,
                    best_effort_cache_credit_tokens=evidence.best_effort_cache_credit_tokens,
                    global_waiting_compute_tokens=waiting_compute,
                    global_waiting_best_effort_cache_credit_tokens=waiting_credit,
                    pending_compute_tokens=pending_compute,
                    pending_best_effort_cache_credit_tokens=pending_credit,
                    running_request_blocks=running_blocks,
                )
            )
        if not candidates:
            raise TarsCapacityError("no active TARS target has sufficient capacity")
        selection = select_tars_candidate(spec.request, candidates, self._config)
        if selection is None:
            raise TarsCapacityError("no selectable TARS target")
        return selection.primary_target

    def _can_reserve_locked(
        self,
        target: TarsTarget,
        evidence: TarsCacheEvidence,
        reserved_blocks: int,
    ) -> bool:
        replica = self._replicas.get(target)
        if replica is None or not replica.active:
            return False
        pending_compute = sum(self._workloads[key].compute_tokens for key in replica.pending if key in self._workloads)
        used_blocks = sum(
            self._workloads[key].reserved_blocks for key in replica.pending | replica.running if key in self._workloads
        )
        return (
            pending_compute + evidence.estimated_compute_tokens <= replica.spec.pending_token_capacity
            and used_blocks + reserved_blocks <= replica.spec.running_block_capacity
        )

    def _remove_queue_entry_locked(self, target: TarsTarget, key: TarsAttemptKey) -> None:
        queue = self._queues.get(target)
        if queue is None or key not in queue:
            raise TarsStateError("attempt queue ownership is missing")
        queue.remove(key)

    def _finish_attempt_locked(self, key: TarsAttemptKey) -> None:
        attempt = self._attempts.pop(key)
        if attempt.phase is TarsAttemptPhase.QUEUED:
            self._remove_queue_entry_locked(attempt.selected_target, key)
        workload = self._workloads.pop(key, None)
        self._reservations.pop(key, None)
        if workload is not None:
            replica = self._replicas[workload.target]
            replica.pending.discard(key)
            replica.running.discard(key)

    def _prune_inactive_replicas_locked(self) -> None:
        referenced = {
            target
            for attempt in self._attempts.values()
            for target in (attempt.selected_target, attempt.final_target, attempt.owner_target)
            if target is not None
        }
        referenced.update(self._reservations.values())
        for target in tuple(self._replicas):
            replica = self._replicas[target]
            if replica.active or target in referenced or replica.pending or replica.running:
                continue
            self._replicas.pop(target)
            self._queues.pop(target, None)

    def _snapshot_attempt_locked(self, attempt: _AttemptState) -> TarsAttemptSnapshot:
        return TarsAttemptSnapshot(
            key=attempt.spec.key,
            phase=attempt.phase,
            selected_target=attempt.selected_target,
            final_target=attempt.final_target,
            owner_target=attempt.owner_target,
            topology_epoch=attempt.topology_epoch,
            queued_at=attempt.queued_at,
            visited_targets=tuple(attempt.visited_targets),
        )

    def _snapshot_replica_locked(
        self,
        target: TarsTarget,
        replica: _ReplicaState,
    ) -> TarsReplicaSnapshot:
        pending_workloads = [self._workloads[key] for key in replica.pending]
        pending_compute = sum(workload.compute_tokens for workload in pending_workloads)
        reserved_blocks = sum(self._workloads[key].reserved_blocks for key in replica.pending | replica.running)
        now = float(self._clock())
        pending_remaining = round(
            sum(
                max(workload.estimated_ttft - max(0.0, now - workload.reserved_at), 0.0)
                for workload in pending_workloads
            ),
            4,
        )
        return TarsReplicaSnapshot(
            target=target,
            active=replica.active,
            pending_token_capacity=replica.spec.pending_token_capacity,
            pending_compute_tokens=pending_compute,
            pending_budget_remaining=max(replica.spec.pending_token_capacity - pending_compute, 0),
            running_block_capacity=replica.spec.running_block_capacity,
            reserved_blocks=reserved_blocks,
            pending_ttft_remaining=pending_remaining,
            last_ttft=replica.last_ttft,
        )

    @staticmethod
    def _snapshot_workload(workload: _WorkloadState) -> TarsWorkloadSnapshot:
        return TarsWorkloadSnapshot(
            key=workload.key,
            target=workload.target,
            phase=workload.phase,
            compute_tokens=workload.compute_tokens,
            cache_credit_tokens=workload.cache_credit_tokens,
            reserved_blocks=workload.reserved_blocks,
            reserved_at=workload.reserved_at,
            estimated_ttft=workload.estimated_ttft,
        )

    def _assert_invariants_locked(self) -> None:
        queue_counts: dict[TarsAttemptKey, int] = {}
        for target, keys in self._queues.items():
            if target not in self._replicas:
                raise RuntimeError("TARS queue targets an unknown replica")
            for key in keys:
                queue_counts[key] = queue_counts.get(key, 0) + 1
                attempt = self._attempts.get(key)
                if attempt is None or attempt.selected_target != target:
                    raise RuntimeError("TARS queue and attempt target disagree")
        if any(count != 1 for count in queue_counts.values()):
            raise RuntimeError("TARS attempt appears in multiple queues")

        owned_keys = {
            key
            for key, attempt in self._attempts.items()
            if attempt.phase in (TarsAttemptPhase.RESERVED, TarsAttemptPhase.RUNNING)
        }
        if set(self._reservations) != owned_keys or set(self._workloads) != owned_keys:
            raise RuntimeError("TARS reservation/workload ownership mismatch")
        if set(queue_counts) != set(self._attempts) - owned_keys:
            raise RuntimeError("TARS queued-attempt index mismatch")

        replica_pending_counts: dict[TarsAttemptKey, int] = {}
        replica_running_counts: dict[TarsAttemptKey, int] = {}
        for replica in self._replicas.values():
            if replica.pending & replica.running:
                raise RuntimeError("TARS workload is both pending and running")
            for key in replica.pending:
                replica_pending_counts[key] = replica_pending_counts.get(key, 0) + 1
            for key in replica.running:
                replica_running_counts[key] = replica_running_counts.get(key, 0) + 1
        if any(count != 1 for count in replica_pending_counts.values()) or any(
            count != 1 for count in replica_running_counts.values()
        ):
            raise RuntimeError("TARS workload appears in multiple replica indexes")
        expected_pending = {
            key for key, workload in self._workloads.items() if workload.phase is TarsAttemptPhase.RESERVED
        }
        expected_running = {
            key for key, workload in self._workloads.items() if workload.phase is TarsAttemptPhase.RUNNING
        }
        if set(replica_pending_counts) != expected_pending or set(replica_running_counts) != expected_running:
            raise RuntimeError("TARS replica workload index mismatch")

        for key in owned_keys:
            attempt = self._attempts[key]
            target = self._reservations[key]
            workload = self._workloads[key]
            if (
                attempt.owner_target != target
                or attempt.final_target != target
                or workload.target != target
                or attempt.phase is not workload.phase
                or target not in self._replicas
            ):
                raise RuntimeError("TARS attempt has inconsistent final ownership")
