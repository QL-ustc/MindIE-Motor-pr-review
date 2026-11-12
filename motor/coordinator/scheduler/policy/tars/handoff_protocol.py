# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Dormant reliability protocol for TARS Handoff scheduling.

The protocol models Scheduler and Router ownership boundaries without changing
Motor's transport, scheduling-policy registry, or request dispatch path.
"""

import hashlib
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from motor.coordinator.scheduler.policy.tars.original.models import TarsTarget
from motor.coordinator.scheduler.policy.tars.state_runtime import (
    TarsAttemptKey,
    TarsAttemptSpec,
    TarsCapacityError,
    TarsStaleTopologyError,
    TarsStateError,
    TarsStateRuntime,
)


class HandoffAttemptState(str, Enum):
    ADMITTED = "admitted"
    QUEUED = "queued"
    TARGET_RESERVED = "target_reserved"
    WORKLOAD_COMMITTED = "workload_committed"
    DELIVERY_PENDING = "delivery_pending"
    OWNERSHIP_TRANSFERRED = "ownership_transferred"
    PREFILL_SETTLING = "prefill_settling"
    PREFILL_COMPLETED = "prefill_completed"
    REQUEST_COMPLETED = "request_completed"
    CANCELLED = "cancelled"
    RESERVATION_RELEASED = "reservation_released"
    ALLOCATION_REVOKING = "allocation_revoking"
    ALLOCATION_REVOKED = "allocation_revoked"
    WORKLOAD_RELEASING = "workload_releasing"
    ATTEMPT_ABORTED = "attempt_aborted"


class HandoffRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    UNIFIED = "unified"


class HandoffLifecycleEvent(str, Enum):
    PREFILL_COMPLETE = "prefill_complete"
    REQUEST_COMPLETE = "request_complete"
    ABORT = "abort"


class HandoffServerState(str, Enum):
    RUNNING = "running"
    DRAINING = "draining"
    STOPPED = "stopped"
    RECOVERING = "recovering"


class HandoffStartupMode(str, Enum):
    FRESH = "fresh"
    GRACEFUL = "graceful"
    UNGRACEFUL = "ungraceful"


class HandoffFailureCode(str, Enum):
    STALE_SCHEDULER_EPOCH = "stale_scheduler_epoch"
    STALE_TOPOLOGY = "stale_topology"
    ATTEMPT_CONFLICT = "attempt_conflict"
    ATTEMPT_NOT_FOUND = "attempt_not_found"
    STALE_ATTEMPT = "stale_attempt"
    INVALID_STATE = "invalid_state"
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    DELIVERY_TOKEN_MISMATCH = "delivery_token_mismatch"  # nosec B105 -- protocol error code
    PRECONFIRM_DISPATCH = "preconfirm_dispatch"
    ROUTER_LEDGER_CONFLICT = "router_ledger_conflict"
    TARGET_MISMATCH = "target_mismatch"
    ENGINE_TYPE_MISMATCH = "engine_type_mismatch"
    CIRCUIT_OPEN = "circuit_open"
    SERVER_TIMEOUT = "server_timeout"
    CLIENT_TIMEOUT = "client_timeout"
    DELIVERY_LEASE_EXPIRED = "delivery_lease_expired"
    SERVER_DRAINING = "server_draining"
    SERVER_NOT_READY = "server_not_ready"
    BOUNDARY_FAILURE = "boundary_failure"
    INTERNAL_STATE = "internal_state"


class HandoffFaultPoint(str, Enum):
    AFTER_ENQUEUE = "after_enqueue"
    AFTER_TARGET_RESERVATION = "after_target_reservation"
    AFTER_WORKLOAD_COMMIT = "after_workload_commit"
    AFTER_DELIVERY_PREPARE = "after_delivery_prepare"
    AFTER_ROUTER_PERSIST = "after_router_persist"
    AFTER_CONFIRM = "after_confirm"
    AFTER_PREFILL_RELEASE = "after_prefill_release"
    BEFORE_DECODE_COMMIT = "before_decode_commit"


class HandoffBoundaryFailure(RuntimeError):
    """Injected or observed failure at a protocol ownership boundary."""


class _HandoffServerTimeout(RuntimeError):
    """Internal signal used to compensate an expired server operation."""


def _positive_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return float(value)


@dataclass(frozen=True)
class HandoffTimeoutBudget:
    """Ordered timeout budget for the dormant Scheduler/Router boundary."""

    queue_wait_s: float = 1.0
    final_query_s: float = 1.0
    exact_commit_s: float = 1.0
    settlement_margin_s: float = 1.0
    server_timeout_s: float = 5.0
    client_timeout_s: float = 6.0
    delivery_lease_s: float = 10.0
    drain_timeout_s: float = 30.0
    recovery_cooldown_s: float = 30.0

    def __post_init__(self) -> None:
        fields = (
            "queue_wait_s",
            "final_query_s",
            "exact_commit_s",
            "settlement_margin_s",
            "server_timeout_s",
            "client_timeout_s",
            "delivery_lease_s",
            "drain_timeout_s",
            "recovery_cooldown_s",
        )
        for field_name in fields:
            object.__setattr__(
                self,
                field_name,
                _positive_number(getattr(self, field_name), field_name),
            )
        stage_total = self.queue_wait_s + self.final_query_s + self.exact_commit_s + self.settlement_margin_s
        if self.server_timeout_s <= stage_total:
            raise ValueError("server_timeout_s must exceed the stage budget plus settlement margin")
        if self.client_timeout_s <= self.server_timeout_s:
            raise ValueError("client_timeout_s must exceed server_timeout_s")


@dataclass(frozen=True, order=True)
class HandoffAttemptId:
    request_id: str
    attempt_seq: int
    scheduler_epoch: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.attempt_seq <= 0:
            raise ValueError("attempt_seq must be positive")
        if self.scheduler_epoch < 0:
            raise ValueError("scheduler_epoch must be non-negative")

    @property
    def key(self) -> TarsAttemptKey:
        return TarsAttemptKey(self.request_id, self.attempt_seq)


@dataclass(frozen=True)
class HandoffRecoveryEvidence:
    """Operator-supplied evidence required to leave cold recovery isolation."""

    previous_scheduler_epoch: int
    router_entries: tuple[HandoffAttemptId, ...]
    operator_confirmed: bool

    def __post_init__(self) -> None:
        if isinstance(self.previous_scheduler_epoch, bool) or self.previous_scheduler_epoch < 0:
            raise ValueError("previous_scheduler_epoch must be non-negative")
        if not isinstance(self.operator_confirmed, bool):
            raise ValueError("operator_confirmed must be a boolean")


class HandoffAttemptSequencer:
    """Allocate fresh attempt identities within one Scheduler epoch."""

    def __init__(self, root_request_id: str, scheduler_epoch: int) -> None:
        if not root_request_id:
            raise ValueError("root_request_id must be non-empty")
        if scheduler_epoch < 0:
            raise ValueError("scheduler_epoch must be non-negative")
        self._root_request_id = root_request_id
        self._scheduler_epoch = scheduler_epoch
        self._next_attempt_seq = 1
        self._lock = threading.Lock()

    def reserve(self) -> HandoffAttemptId:
        with self._lock:
            identity = HandoffAttemptId(
                self._root_request_id,
                self._next_attempt_seq,
                self._scheduler_epoch,
            )
            self._next_attempt_seq += 1
            return identity


@dataclass(frozen=True)
class HandoffTargetSpec:
    target: TarsTarget
    role: HandoffRole
    engine_type: str
    circuit_open: bool = False

    def __post_init__(self) -> None:
        if not self.engine_type:
            raise ValueError("engine_type must be non-empty")


@dataclass(frozen=True)
class HandoffAllocateRequest:
    identity: HandoffAttemptId
    attempt_spec: TarsAttemptSpec
    scheduler_epoch: int
    expected_topology_epoch: int
    role: HandoffRole = HandoffRole.PREFILL


@dataclass(frozen=True)
class HandoffConfirmRequest:
    identity: HandoffAttemptId
    delivery_token: str
    scheduler_epoch: int


@dataclass(frozen=True)
class HandoffRevokeRequest:
    identity: HandoffAttemptId
    delivery_token: str
    scheduler_epoch: int


@dataclass(frozen=True)
class HandoffStatusRequest:
    identity: HandoffAttemptId
    scheduler_epoch: int


@dataclass(frozen=True)
class HandoffDecodeCommitRequest:
    identity: HandoffAttemptId
    target: TarsTarget
    scheduler_epoch: int
    expected_topology_epoch: int


@dataclass(frozen=True)
class HandoffLifecycleRequest:
    identity: HandoffAttemptId
    event: HandoffLifecycleEvent
    scheduler_epoch: int
    observed_ttft: float | None = None


@dataclass(frozen=True)
class HandoffFailure:
    code: HandoffFailureCode
    message: str


@dataclass(frozen=True)
class HandoffAllocation:
    identity: HandoffAttemptId
    target: TarsTarget
    delivery_token: str
    topology_epoch: int
    engine_type: str
    role: HandoffRole = HandoffRole.PREFILL
    confirmed: bool = False
    dispatched: bool = False
    delivery_expires_at: float | None = None


@dataclass(frozen=True)
class HandoffAttemptSnapshot:
    identity: HandoffAttemptId
    state: HandoffAttemptState
    target: TarsTarget
    topology_epoch: int
    engine_type: str
    role: HandoffRole
    decode_target: TarsTarget | None = None


@dataclass(frozen=True)
class HandoffProtocolResult:
    failure: HandoffFailure | None = None
    allocation: HandoffAllocation | None = None
    attempt: HandoffAttemptSnapshot | None = None

    @property
    def ok(self) -> bool:
        return self.failure is None


@dataclass(frozen=True)
class HandoffServerSnapshot:
    scheduler_epoch: int
    state: HandoffServerState
    accepting_admissions: bool
    drain_deadline: float | None
    recovery_started_at: float | None
    active_attempts: tuple[HandoffAttemptSnapshot, ...]
    pending_deliveries: tuple[HandoffAllocation, ...]
    decode_workloads: tuple[tuple[HandoffAttemptId, TarsTarget], ...]
    terminal_receipts: tuple[HandoffAttemptSnapshot, ...]
    target_inventory: tuple[HandoffTargetSpec, ...]


@dataclass(frozen=True)
class HandoffRouterSnapshot:
    entries: tuple[HandoffAllocation, ...]


@dataclass
class _AttemptRecord:
    request: HandoffAllocateRequest
    state: HandoffAttemptState
    target: TarsTarget
    engine_type: str
    delivery_token: str = ""
    delivery_expires_at: float | None = None
    prefill_released: bool = False
    decode_target: TarsTarget | None = None


FaultHook = Callable[[HandoffFaultPoint, HandoffAttemptId], None]


def _success(
    *,
    allocation: HandoffAllocation | None = None,
    attempt: HandoffAttemptSnapshot | None = None,
) -> HandoffProtocolResult:
    return HandoffProtocolResult(allocation=allocation, attempt=attempt)


def _failure(code: HandoffFailureCode, message: str) -> HandoffProtocolResult:
    return HandoffProtocolResult(failure=HandoffFailure(code, message))


class HandoffProtocolServer:
    """Scheduler-side owner of dormant single-role admission and compensation."""

    def __init__(
        self,
        runtime: TarsStateRuntime,
        scheduler_epoch: int,
        fault_hook: FaultHook | None = None,
        clock: Callable[[], float] | None = None,
        timeout_budget: HandoffTimeoutBudget | None = None,
        startup_mode: HandoffStartupMode = HandoffStartupMode.FRESH,
    ) -> None:
        if scheduler_epoch < 0:
            raise ValueError("scheduler_epoch must be non-negative")
        if not isinstance(startup_mode, HandoffStartupMode):
            raise ValueError("startup_mode must be a HandoffStartupMode")
        self._runtime = runtime
        self._scheduler_epoch = scheduler_epoch
        self._fault_hook = fault_hook
        self._clock = clock or time.monotonic
        self._timeout_budget = timeout_budget or HandoffTimeoutBudget()
        self._lock = threading.RLock()
        self._targets: dict[TarsTarget, HandoffTargetSpec] = {}
        self._attempts: dict[HandoffAttemptId, _AttemptRecord] = {}
        self._pending_deliveries: dict[HandoffAttemptId, HandoffAllocation] = {}
        self._decode_workloads: dict[HandoffAttemptId, TarsTarget] = {}
        self._terminal_receipts: dict[HandoffAttemptId, HandoffAttemptSnapshot] = {}
        self._terminal_tokens: dict[HandoffAttemptId, str] = {}
        self._terminal_failures: dict[HandoffAttemptId, HandoffFailure] = {}
        self._drain_deadline: float | None = None
        self._recovery_started_at: float | None = None
        self._state = HandoffServerState.RUNNING
        if startup_mode is HandoffStartupMode.UNGRACEFUL:
            self._runtime.quiesce()
            self._state = HandoffServerState.RECOVERING
            self._recovery_started_at = self._now()
        elif startup_mode is HandoffStartupMode.GRACEFUL:
            self._require_empty_runtime("graceful startup")

    @property
    def timeout_budget(self) -> HandoffTimeoutBudget:
        return self._timeout_budget

    @property
    def clock(self) -> Callable[[], float]:
        return self._clock

    def sync_targets(self, topology_epoch: int, targets: tuple[HandoffTargetSpec, ...]) -> bool:
        with self._lock:
            if topology_epoch != self._runtime.topology_epoch:
                return False
            target_ids = [item.target for item in targets]
            if len(target_ids) != len(set(target_ids)):
                raise ValueError("Handoff targets must be unique")
            known_targets = {item.target for item in self._runtime.snapshot().replicas}
            if set(target_ids) != known_targets:
                raise ValueError("Handoff target inventory must match TARS runtime topology")
            self._targets = {item.target: item for item in targets}
            return True

    def allocate(self, request: HandoffAllocateRequest) -> HandoffProtocolResult:
        with self._lock:
            self._poll_locked()
            invalid = self._validate_allocate_request(request)
            if invalid is not None:
                return invalid
            existing = self._attempts.get(request.identity)
            if existing is not None:
                if existing.request != request:
                    return _failure(HandoffFailureCode.ATTEMPT_CONFLICT, "attempt payload changed")
                return _success(
                    allocation=self._allocation(
                        existing,
                        confirmed=existing.state is not HandoffAttemptState.DELIVERY_PENDING,
                    ),
                    attempt=self._snapshot_attempt(existing),
                )
            if request.identity in self._terminal_receipts:
                return _failure(HandoffFailureCode.ATTEMPT_CONFLICT, "attempt identity is terminal")
            unavailable = self._admission_failure()
            if unavailable is not None:
                return unavailable

            operation_started_at = self._now()
            record = _AttemptRecord(
                request=request,
                state=HandoffAttemptState.ADMITTED,
                target=request.attempt_spec.evidence[0].target,
                engine_type="",
            )
            self._attempts[request.identity] = record
            try:
                self._allocate_stages(record, operation_started_at)
                return _success(
                    allocation=self._pending_deliveries[request.identity],
                    attempt=self._snapshot_attempt(record),
                )
            except HandoffBoundaryFailure as exc:
                self._compensate_allocation(record)
                return _failure(HandoffFailureCode.BOUNDARY_FAILURE, str(exc))
            except _HandoffServerTimeout as exc:
                self._compensate_allocation(record)
                return _failure(HandoffFailureCode.SERVER_TIMEOUT, str(exc))
            except TarsStaleTopologyError as exc:
                self._compensate_allocation(record)
                return _failure(HandoffFailureCode.STALE_TOPOLOGY, str(exc))
            except TarsCapacityError as exc:
                self._compensate_allocation(record)
                return _failure(HandoffFailureCode.CAPACITY_EXHAUSTED, str(exc))
            except TarsStateError as exc:
                self._compensate_allocation(record)
                return _failure(HandoffFailureCode.INTERNAL_STATE, str(exc))
            except BaseException:
                self._compensate_allocation(record)
                raise

    def confirm(self, request: HandoffConfirmRequest) -> HandoffProtocolResult:
        with self._lock:
            operation_started_at = self._now()
            invalid = self._validate_epoch(request.identity, request.scheduler_epoch)
            if invalid is not None:
                return invalid
            self._expire_pending_deliveries_locked()
            record = self._attempts.get(request.identity)
            if record is None:
                return self._terminal_or_missing(request.identity)
            if request.delivery_token != record.delivery_token:
                return _failure(
                    HandoffFailureCode.DELIVERY_TOKEN_MISMATCH,
                    "delivery confirmation token does not match",
                )
            if record.state is HandoffAttemptState.OWNERSHIP_TRANSFERRED:
                return _success(
                    allocation=self._allocation(record, confirmed=True),
                    attempt=self._snapshot_attempt(record),
                )
            if record.state is not HandoffAttemptState.DELIVERY_PENDING:
                return _failure(
                    HandoffFailureCode.INVALID_STATE,
                    "attempt is not awaiting delivery",
                )
            record.state = HandoffAttemptState.OWNERSHIP_TRANSFERRED
            self._pending_deliveries.pop(request.identity, None)
            try:
                self._run_fault(HandoffFaultPoint.AFTER_CONFIRM, request.identity)
                self._check_server_timeout(operation_started_at, "delivery confirmation")
            except HandoffBoundaryFailure as exc:
                self._abort_record(record)
                return _failure(HandoffFailureCode.BOUNDARY_FAILURE, str(exc))
            except _HandoffServerTimeout as exc:
                self._abort_record(record)
                return _failure(HandoffFailureCode.SERVER_TIMEOUT, str(exc))
            except BaseException:
                self._abort_record(record)
                raise
            return _success(
                allocation=self._allocation(record, confirmed=True),
                attempt=self._snapshot_attempt(record),
            )

    def revoke(self, request: HandoffRevokeRequest) -> HandoffProtocolResult:
        with self._lock:
            invalid = self._validate_epoch(request.identity, request.scheduler_epoch)
            if invalid is not None:
                return invalid
            self._expire_pending_deliveries_locked()
            terminal = self._terminal_receipts.get(request.identity)
            if terminal is not None:
                if terminal.state is HandoffAttemptState.ALLOCATION_REVOKED:
                    if request.delivery_token != self._terminal_tokens[request.identity]:
                        return _failure(
                            HandoffFailureCode.DELIVERY_TOKEN_MISMATCH,
                            "delivery revocation token does not match",
                        )
                    return _success(attempt=terminal)
                return _failure(HandoffFailureCode.INVALID_STATE, "attempt is already terminal")
            record = self._attempts.get(request.identity)
            if record is None:
                return _failure(HandoffFailureCode.ATTEMPT_NOT_FOUND, "attempt does not exist")
            if request.delivery_token != record.delivery_token:
                return _failure(
                    HandoffFailureCode.DELIVERY_TOKEN_MISMATCH,
                    "delivery revocation token does not match",
                )
            if record.state is not HandoffAttemptState.DELIVERY_PENDING:
                return _failure(
                    HandoffFailureCode.INVALID_STATE,
                    "confirmed ownership cannot be revoked",
                )
            record.state = HandoffAttemptState.ALLOCATION_REVOKING
            self._runtime.abort(request.identity.key)
            self._pending_deliveries.pop(request.identity, None)
            return self._finish_record(record, HandoffAttemptState.ALLOCATION_REVOKED)

    def status(self, request: HandoffStatusRequest) -> HandoffProtocolResult:
        with self._lock:
            invalid = self._validate_epoch(request.identity, request.scheduler_epoch)
            if invalid is not None:
                return invalid
            self._expire_pending_deliveries_locked()
            record = self._attempts.get(request.identity)
            if record is not None:
                allocation = None
                if record.delivery_token:
                    allocation = self._allocation(
                        record,
                        confirmed=record.state is not HandoffAttemptState.DELIVERY_PENDING,
                    )
                return _success(allocation=allocation, attempt=self._snapshot_attempt(record))
            return self._terminal_or_missing(request.identity)

    def lifecycle(self, request: HandoffLifecycleRequest) -> HandoffProtocolResult:
        with self._lock:
            invalid = self._validate_epoch(request.identity, request.scheduler_epoch)
            if invalid is not None:
                return invalid
            terminal = self._terminal_receipts.get(request.identity)
            if terminal is not None:
                if self._terminal_matches_event(terminal.state, request.event):
                    return _success(attempt=terminal)
                return _failure(HandoffFailureCode.INVALID_STATE, "attempt is already terminal")
            record = self._attempts.get(request.identity)
            if record is None:
                return _failure(HandoffFailureCode.ATTEMPT_NOT_FOUND, "attempt does not exist")
            if request.event is HandoffLifecycleEvent.PREFILL_COMPLETE:
                return self._settle_prefill(record, request.observed_ttft)
            if request.event is HandoffLifecycleEvent.REQUEST_COMPLETE:
                return self._complete_request(record)
            return self._abort_record(record)

    def commit_decode(self, request: HandoffDecodeCommitRequest) -> HandoffProtocolResult:
        with self._lock:
            invalid = self._validate_epoch(request.identity, request.scheduler_epoch)
            if invalid is not None:
                return invalid
            if request.expected_topology_epoch != self._runtime.topology_epoch:
                return _failure(HandoffFailureCode.STALE_TOPOLOGY, "decode topology epoch is stale")
            record = self._attempts.get(request.identity)
            if record is None:
                return self._terminal_or_missing(request.identity)
            if record.request.role is not HandoffRole.PREFILL:
                return _failure(
                    HandoffFailureCode.INVALID_STATE,
                    "unified attempts do not bind Decode",
                )
            if record.state is not HandoffAttemptState.PREFILL_COMPLETED:
                return _failure(
                    HandoffFailureCode.INVALID_STATE,
                    "decode requires completed prefill",
                )
            existing = self._decode_workloads.get(request.identity)
            if existing is not None:
                if existing != request.target:
                    return _failure(HandoffFailureCode.TARGET_MISMATCH, "decode target changed")
                return _success(attempt=self._snapshot_attempt(record))
            target = self._targets.get(request.target)
            if target is None or target.role is not HandoffRole.DECODE:
                return _failure(
                    HandoffFailureCode.TARGET_MISMATCH,
                    "target is not a decode replica",
                )
            if target.circuit_open:
                return _failure(HandoffFailureCode.CIRCUIT_OPEN, "decode target circuit is open")
            if target.engine_type != record.engine_type:
                return _failure(
                    HandoffFailureCode.ENGINE_TYPE_MISMATCH,
                    "P and D engine types differ",
                )
            try:
                self._run_fault(HandoffFaultPoint.BEFORE_DECODE_COMMIT, request.identity)
            except HandoffBoundaryFailure as exc:
                return _failure(HandoffFailureCode.BOUNDARY_FAILURE, str(exc))
            except BaseException:
                self._abort_record(record)
                raise
            self._decode_workloads[request.identity] = request.target
            record.decode_target = request.target
            self._assert_invariants_locked()
            return _success(attempt=self._snapshot_attempt(record))

    def quiesce(self) -> None:
        with self._lock:
            for record in tuple(self._attempts.values()):
                self._abort_record(record)
            self._pending_deliveries.clear()
            self._decode_workloads.clear()
            self._terminal_receipts.clear()
            self._terminal_tokens.clear()
            self._terminal_failures.clear()
            self._runtime.quiesce()
            self._state = HandoffServerState.STOPPED
            self._drain_deadline = None
            self._assert_invariants_locked()

    def begin_drain(self, timeout_s: float | None = None) -> bool:
        """Stop new admissions and let active ownership finish until a deadline."""
        with self._lock:
            if self._state is HandoffServerState.RECOVERING:
                return False
            if self._state is HandoffServerState.STOPPED:
                return True
            timeout = self._timeout_budget.drain_timeout_s
            if timeout_s is not None:
                timeout = _positive_number(timeout_s, "timeout_s")
            self._state = HandoffServerState.DRAINING
            self._drain_deadline = self._now() + timeout
            self._maybe_stop_draining_locked()
            return True

    def poll(self) -> HandoffServerState:
        """Advance lease and drain timers without enabling recovery automatically."""
        with self._lock:
            self._poll_locked()
            return self._state

    def complete_recovery(self, evidence: HandoffRecoveryEvidence) -> HandoffProtocolResult:
        """Leave cold isolation only after cooldown and explicit Router reconciliation."""
        with self._lock:
            if self._state is not HandoffServerState.RECOVERING:
                return _failure(HandoffFailureCode.INVALID_STATE, "server is not recovering")
            assert self._recovery_started_at is not None
            cooldown_elapsed = self._now() - self._recovery_started_at
            if cooldown_elapsed < self._timeout_budget.recovery_cooldown_s:
                return _failure(HandoffFailureCode.SERVER_NOT_READY, "recovery cooldown is active")
            if evidence.previous_scheduler_epoch >= self._scheduler_epoch:
                return _failure(
                    HandoffFailureCode.SERVER_NOT_READY,
                    "previous Scheduler epoch is invalid",
                )
            if evidence.router_entries:
                return _failure(
                    HandoffFailureCode.SERVER_NOT_READY,
                    "Router ledger is not reconciled",
                )
            if not evidence.operator_confirmed:
                return _failure(
                    HandoffFailureCode.SERVER_NOT_READY,
                    "operator confirmation is required",
                )
            try:
                self._require_empty_runtime("recovery completion")
            except RuntimeError as exc:
                return _failure(HandoffFailureCode.SERVER_NOT_READY, str(exc))
            self._state = HandoffServerState.RUNNING
            self._recovery_started_at = None
            return _success()

    def snapshot(self) -> HandoffServerSnapshot:
        with self._lock:
            self._poll_locked()
            return HandoffServerSnapshot(
                scheduler_epoch=self._scheduler_epoch,
                state=self._state,
                accepting_admissions=self._state is HandoffServerState.RUNNING,
                drain_deadline=self._drain_deadline,
                recovery_started_at=self._recovery_started_at,
                active_attempts=tuple(self._snapshot_attempt(record) for _, record in sorted(self._attempts.items())),
                pending_deliveries=tuple(allocation for _, allocation in sorted(self._pending_deliveries.items())),
                decode_workloads=tuple(sorted(self._decode_workloads.items())),
                terminal_receipts=tuple(receipt for _, receipt in sorted(self._terminal_receipts.items())),
                target_inventory=tuple(spec for _, spec in sorted(self._targets.items())),
            )

    def assert_invariants(self) -> None:
        with self._lock:
            self._assert_invariants_locked()
            self._runtime.assert_invariants()

    def _allocate_stages(self, record: _AttemptRecord, operation_started_at: float) -> None:
        request = record.request
        stage_started_at = self._now()
        runtime_attempt = self._runtime.enqueue(request.attempt_spec)
        record.state = HandoffAttemptState.QUEUED
        record.target = runtime_attempt.selected_target
        self._run_fault(HandoffFaultPoint.AFTER_ENQUEUE, request.identity)
        self._check_stage_timeout(stage_started_at, self._timeout_budget.queue_wait_s, "queue wait")
        stage_started_at = self._now()
        try:
            runtime_attempt = self._runtime.lock_final_target(
                request.identity.key,
                expected_topology_epoch=request.expected_topology_epoch,
            )
        except TarsCapacityError:
            self._runtime.rebalance(
                request.identity.key,
                expected_topology_epoch=request.expected_topology_epoch,
            )
            runtime_attempt = self._runtime.lock_final_target(
                request.identity.key,
                expected_topology_epoch=request.expected_topology_epoch,
            )
        record.state = HandoffAttemptState.TARGET_RESERVED
        record.target = runtime_attempt.final_target or runtime_attempt.selected_target
        target = self._targets.get(record.target)
        if target is None or target.role is not request.role:
            raise TarsStateError("selected target does not match the requested role")
        record.engine_type = target.engine_type
        self._run_fault(HandoffFaultPoint.AFTER_TARGET_RESERVATION, request.identity)
        self._check_stage_timeout(
            stage_started_at,
            self._timeout_budget.final_query_s,
            "final target query",
        )
        stage_started_at = self._now()
        self._runtime.reserve(
            request.identity.key,
            expected_topology_epoch=request.expected_topology_epoch,
        )
        record.state = HandoffAttemptState.WORKLOAD_COMMITTED
        self._run_fault(HandoffFaultPoint.AFTER_WORKLOAD_COMMIT, request.identity)
        self._check_stage_timeout(
            stage_started_at,
            self._timeout_budget.exact_commit_s,
            "exact workload commit",
        )
        record.delivery_token = self._delivery_token(record)
        record.delivery_expires_at = self._now() + self._timeout_budget.delivery_lease_s
        record.state = HandoffAttemptState.DELIVERY_PENDING
        self._pending_deliveries[request.identity] = self._allocation(record, confirmed=False)
        self._run_fault(HandoffFaultPoint.AFTER_DELIVERY_PREPARE, request.identity)
        self._check_server_timeout(operation_started_at, "allocation")
        self._assert_invariants_locked()

    def _settle_prefill(
        self,
        record: _AttemptRecord,
        observed_ttft: float | None,
    ) -> HandoffProtocolResult:
        if record.request.role is not HandoffRole.PREFILL:
            return _failure(
                HandoffFailureCode.INVALID_STATE,
                "unified attempts have no Prefill boundary",
            )
        if record.state is HandoffAttemptState.PREFILL_COMPLETED:
            return _success(attempt=self._snapshot_attempt(record))
        if record.state not in (
            HandoffAttemptState.OWNERSHIP_TRANSFERRED,
            HandoffAttemptState.PREFILL_SETTLING,
        ):
            return _failure(
                HandoffFailureCode.INVALID_STATE,
                "prefill ownership was not transferred",
            )
        record.state = HandoffAttemptState.PREFILL_SETTLING
        if not record.prefill_released:
            released = self._runtime.complete_prefill(
                record.request.identity.key,
                observed_ttft,
                keep_running=False,
            )
            if not released:
                return _failure(HandoffFailureCode.INTERNAL_STATE, "prefill workload is missing")
            record.prefill_released = True
            try:
                self._run_fault(HandoffFaultPoint.AFTER_PREFILL_RELEASE, record.request.identity)
            except HandoffBoundaryFailure as exc:
                return _failure(HandoffFailureCode.BOUNDARY_FAILURE, str(exc))
            except BaseException:
                self._abort_record(record)
                raise
        record.state = HandoffAttemptState.PREFILL_COMPLETED
        self._assert_invariants_locked()
        return _success(attempt=self._snapshot_attempt(record))

    def _complete_request(self, record: _AttemptRecord) -> HandoffProtocolResult:
        if record.request.role is HandoffRole.UNIFIED:
            if record.state is not HandoffAttemptState.OWNERSHIP_TRANSFERRED:
                return _failure(HandoffFailureCode.INVALID_STATE, "unified request is not active")
            if not self._runtime.complete_request(record.request.identity.key):
                return _failure(HandoffFailureCode.INTERNAL_STATE, "unified workload is missing")
            return self._finish_record(record, HandoffAttemptState.REQUEST_COMPLETED)
        if record.state is not HandoffAttemptState.PREFILL_COMPLETED:
            return _failure(HandoffFailureCode.INVALID_STATE, "request is not ready to complete")
        if record.request.identity not in self._decode_workloads:
            return _failure(HandoffFailureCode.INVALID_STATE, "decode workload was not committed")
        self._decode_workloads.pop(record.request.identity, None)
        return self._finish_record(record, HandoffAttemptState.REQUEST_COMPLETED)

    def _abort_record(self, record: _AttemptRecord) -> HandoffProtocolResult:
        record.state = HandoffAttemptState.WORKLOAD_RELEASING
        self._runtime.abort(record.request.identity.key)
        self._pending_deliveries.pop(record.request.identity, None)
        self._decode_workloads.pop(record.request.identity, None)
        return self._finish_record(record, HandoffAttemptState.ATTEMPT_ABORTED)

    def _compensate_allocation(self, record: _AttemptRecord) -> None:
        if record.state is HandoffAttemptState.TARGET_RESERVED:
            record.state = HandoffAttemptState.RESERVATION_RELEASED
            try:
                self._runtime.release_final_target(
                    record.request.identity.key,
                    expected_topology_epoch=record.request.expected_topology_epoch,
                )
            except TarsStateError:
                # An epoch change may invalidate the explicit unlock, while
                # abort remains safe and owns the final cleanup below.
                pass
            record.state = HandoffAttemptState.QUEUED
        elif record.state in (
            HandoffAttemptState.WORKLOAD_COMMITTED,
            HandoffAttemptState.DELIVERY_PENDING,
        ):
            record.state = HandoffAttemptState.ALLOCATION_REVOKING
        else:
            record.state = HandoffAttemptState.CANCELLED
        self._runtime.abort(record.request.identity.key)
        self._pending_deliveries.pop(record.request.identity, None)
        self._finish_record(record, HandoffAttemptState.ALLOCATION_REVOKED)

    def _finish_record(
        self,
        record: _AttemptRecord,
        state: HandoffAttemptState,
        failure: HandoffFailure | None = None,
    ) -> HandoffProtocolResult:
        identity = record.request.identity
        record.state = state
        receipt = self._snapshot_attempt(record)
        self._attempts.pop(identity, None)
        self._terminal_receipts[identity] = receipt
        self._terminal_tokens[identity] = record.delivery_token
        if failure is not None:
            self._terminal_failures[identity] = failure
        self._maybe_stop_draining_locked()
        self._assert_invariants_locked()
        if failure is not None:
            return HandoffProtocolResult(failure=failure, attempt=receipt)
        return _success(attempt=receipt)

    def _validate_allocate_request(
        self,
        request: HandoffAllocateRequest,
    ) -> HandoffProtocolResult | None:
        invalid = self._validate_epoch(request.identity, request.scheduler_epoch)
        if invalid is not None:
            return invalid
        if request.attempt_spec.key != request.identity.key:
            return _failure(HandoffFailureCode.ATTEMPT_CONFLICT, "attempt identity and spec differ")
        if request.role not in (HandoffRole.PREFILL, HandoffRole.UNIFIED):
            return _failure(
                HandoffFailureCode.INVALID_STATE,
                "role is not a single-role admission target",
            )
        if request.expected_topology_epoch != self._runtime.topology_epoch:
            return _failure(HandoffFailureCode.STALE_TOPOLOGY, "allocation topology epoch is stale")
        return None

    def _validate_epoch(
        self,
        identity: HandoffAttemptId,
        scheduler_epoch: int,
    ) -> HandoffProtocolResult | None:
        if scheduler_epoch != self._scheduler_epoch or identity.scheduler_epoch != self._scheduler_epoch:
            return _failure(HandoffFailureCode.STALE_SCHEDULER_EPOCH, "Scheduler epoch is stale")
        return None

    def _terminal_or_missing(self, identity: HandoffAttemptId) -> HandoffProtocolResult:
        terminal = self._terminal_receipts.get(identity)
        if terminal is not None:
            failure = self._terminal_failures.get(identity)
            if failure is not None:
                return HandoffProtocolResult(failure=failure, attempt=terminal)
            return _success(attempt=terminal)
        return _failure(HandoffFailureCode.ATTEMPT_NOT_FOUND, "attempt does not exist")

    def _allocation(self, record: _AttemptRecord, *, confirmed: bool) -> HandoffAllocation:
        return HandoffAllocation(
            identity=record.request.identity,
            target=record.target,
            delivery_token=record.delivery_token,
            topology_epoch=record.request.expected_topology_epoch,
            engine_type=record.engine_type,
            role=record.request.role,
            confirmed=confirmed,
            delivery_expires_at=record.delivery_expires_at,
        )

    @staticmethod
    def _snapshot_attempt(record: _AttemptRecord) -> HandoffAttemptSnapshot:
        return HandoffAttemptSnapshot(
            identity=record.request.identity,
            state=record.state,
            target=record.target,
            topology_epoch=record.request.expected_topology_epoch,
            engine_type=record.engine_type,
            role=record.request.role,
            decode_target=record.decode_target,
        )

    @staticmethod
    def _terminal_matches_event(
        state: HandoffAttemptState,
        event: HandoffLifecycleEvent,
    ) -> bool:
        return (event is HandoffLifecycleEvent.REQUEST_COMPLETE and state is HandoffAttemptState.REQUEST_COMPLETED) or (
            event is HandoffLifecycleEvent.ABORT and state is HandoffAttemptState.ATTEMPT_ABORTED
        )

    @staticmethod
    def _delivery_token(record: _AttemptRecord) -> str:
        identity = record.request.identity
        material = (
            f"{identity.request_id}:{identity.attempt_seq}:{identity.scheduler_epoch}:"
            f"{record.target.instance_id}:{record.target.dp_rank}:"
            f"{record.request.expected_topology_epoch}:{record.request.role.value}"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _run_fault(self, point: HandoffFaultPoint, identity: HandoffAttemptId) -> None:
        if self._fault_hook is not None:
            self._fault_hook(point, identity)

    def _now(self) -> float:
        return float(self._clock())

    def _admission_failure(self) -> HandoffProtocolResult | None:
        if self._state is HandoffServerState.RUNNING:
            return None
        if self._state is HandoffServerState.DRAINING:
            return _failure(HandoffFailureCode.SERVER_DRAINING, "server is draining")
        return _failure(HandoffFailureCode.SERVER_NOT_READY, "server is not accepting admissions")

    def _check_stage_timeout(self, started_at: float, budget_s: float, stage: str) -> None:
        if self._now() - started_at >= budget_s:
            raise _HandoffServerTimeout(f"{stage} exceeded its timeout budget")

    def _check_server_timeout(self, started_at: float, operation: str) -> None:
        if self._now() - started_at >= self._timeout_budget.server_timeout_s:
            raise _HandoffServerTimeout(f"{operation} exceeded server_timeout_s")

    def _expire_pending_deliveries_locked(self) -> None:
        now = self._now()
        expired = tuple(
            identity
            for identity, allocation in self._pending_deliveries.items()
            if allocation.delivery_expires_at is not None and now >= allocation.delivery_expires_at
        )
        for identity in expired:
            record = self._attempts.get(identity)
            if record is None:
                self._pending_deliveries.pop(identity, None)
                continue
            record.state = HandoffAttemptState.ALLOCATION_REVOKING
            self._runtime.abort(identity.key)
            self._pending_deliveries.pop(identity, None)
            failure = HandoffFailure(
                HandoffFailureCode.DELIVERY_LEASE_EXPIRED,
                "delivery lease expired before Router confirmation",
            )
            self._finish_record(record, HandoffAttemptState.ALLOCATION_REVOKED, failure)

    def _poll_locked(self) -> None:
        self._expire_pending_deliveries_locked()
        if self._state is not HandoffServerState.DRAINING:
            return
        self._maybe_stop_draining_locked()
        if self._state is HandoffServerState.STOPPED:
            return
        assert self._drain_deadline is not None
        if self._now() < self._drain_deadline:
            return
        for record in tuple(self._attempts.values()):
            self._abort_record(record)
        self._state = HandoffServerState.STOPPED
        self._drain_deadline = None

    def _maybe_stop_draining_locked(self) -> None:
        if self._state is HandoffServerState.DRAINING and not self._attempts:
            self._state = HandoffServerState.STOPPED
            self._drain_deadline = None

    def _require_empty_runtime(self, context: str) -> None:
        runtime = self._runtime.snapshot()
        if runtime.attempts or runtime.queue_entries or runtime.reservations or runtime.workloads:
            raise RuntimeError(f"{context} requires empty transient runtime state")

    def _assert_invariants_locked(self) -> None:
        if set(self._attempts) & set(self._terminal_receipts):
            raise RuntimeError("Handoff attempt is both active and terminal")
        if not set(self._pending_deliveries) <= set(self._attempts):
            raise RuntimeError("pending delivery has no active attempt")
        if not set(self._decode_workloads) <= set(self._attempts):
            raise RuntimeError("decode workload has no active attempt")
        if not set(self._terminal_failures) <= set(self._terminal_receipts):
            raise RuntimeError("terminal failure has no receipt")
        for identity, allocation in self._pending_deliveries.items():
            record = self._attempts[identity]
            if record.state is not HandoffAttemptState.DELIVERY_PENDING:
                raise RuntimeError("pending delivery and attempt state disagree")
            if allocation.target != record.target or allocation.confirmed:
                raise RuntimeError("pending delivery ownership is inconsistent")
        for identity, target in self._decode_workloads.items():
            record = self._attempts[identity]
            if record.state is not HandoffAttemptState.PREFILL_COMPLETED:
                raise RuntimeError("decode workload exists before prefill completion")
            if record.decode_target != target:
                raise RuntimeError("decode workload target is inconsistent")


class HandoffRouterLedger:
    """Router-side durable-boundary model; dispatch requires confirmation."""

    def __init__(self) -> None:
        self._entries: dict[HandoffAttemptId, HandoffAllocation] = {}
        self._lock = threading.RLock()

    def persist(self, allocation: HandoffAllocation) -> HandoffProtocolResult:
        with self._lock:
            existing = self._entries.get(allocation.identity)
            if existing is not None:
                state_is_not_regressed = (not allocation.confirmed or existing.confirmed == allocation.confirmed) and (
                    not allocation.dispatched or existing.dispatched == allocation.dispatched
                )
                if self._same_delivery(existing, allocation) and state_is_not_regressed:
                    return _success(allocation=existing)
                return _failure(HandoffFailureCode.ROUTER_LEDGER_CONFLICT, "Router entry changed")
            self._entries[allocation.identity] = allocation
            return _success(allocation=allocation)

    def confirm(self, allocation: HandoffAllocation) -> HandoffProtocolResult:
        with self._lock:
            existing = self._entries.get(allocation.identity)
            if existing is None:
                return _failure(HandoffFailureCode.ATTEMPT_NOT_FOUND, "Router entry does not exist")
            if not self._same_delivery(existing, allocation) or not allocation.confirmed:
                return _failure(
                    HandoffFailureCode.ROUTER_LEDGER_CONFLICT,
                    "confirmation changed target",
                )
            confirmed = HandoffAllocation(
                identity=existing.identity,
                target=existing.target,
                delivery_token=existing.delivery_token,
                topology_epoch=existing.topology_epoch,
                engine_type=existing.engine_type,
                role=existing.role,
                confirmed=True,
                dispatched=existing.dispatched,
                delivery_expires_at=existing.delivery_expires_at,
            )
            self._entries[allocation.identity] = confirmed
            return _success(allocation=confirmed)

    def mark_dispatched(self, identity: HandoffAttemptId) -> HandoffProtocolResult:
        with self._lock:
            existing = self._entries.get(identity)
            if existing is None:
                return _failure(HandoffFailureCode.ATTEMPT_NOT_FOUND, "Router entry does not exist")
            if not existing.confirmed:
                return _failure(HandoffFailureCode.PRECONFIRM_DISPATCH, "delivery is not confirmed")
            dispatched = HandoffAllocation(
                identity=existing.identity,
                target=existing.target,
                delivery_token=existing.delivery_token,
                topology_epoch=existing.topology_epoch,
                engine_type=existing.engine_type,
                role=existing.role,
                confirmed=True,
                dispatched=True,
                delivery_expires_at=existing.delivery_expires_at,
            )
            self._entries[identity] = dispatched
            return _success(allocation=dispatched)

    def remove(self, identity: HandoffAttemptId) -> HandoffProtocolResult:
        with self._lock:
            self._entries.pop(identity, None)
            return _success()

    @staticmethod
    def _same_delivery(left: HandoffAllocation, right: HandoffAllocation) -> bool:
        return (
            left.identity,
            left.target,
            left.delivery_token,
            left.topology_epoch,
            left.engine_type,
            left.role,
            left.delivery_expires_at,
        ) == (
            right.identity,
            right.target,
            right.delivery_token,
            right.topology_epoch,
            right.engine_type,
            right.role,
            right.delivery_expires_at,
        )

    def snapshot(self) -> HandoffRouterSnapshot:
        with self._lock:
            return HandoffRouterSnapshot(entries=tuple(entry for _, entry in sorted(self._entries.items())))

    def quiesce(self) -> None:
        with self._lock:
            self._entries.clear()


class HandoffProtocolClient:
    """Client coordinator enforcing Router persistence before Scheduler confirm."""

    def __init__(
        self,
        server: HandoffProtocolServer,
        router_ledger: HandoffRouterLedger,
        fault_hook: FaultHook | None = None,
        clock: Callable[[], float] | None = None,
        timeout_budget: HandoffTimeoutBudget | None = None,
    ) -> None:
        self._server = server
        self._router_ledger = router_ledger
        self._fault_hook = fault_hook
        self._clock = clock or server.clock
        self._timeout_budget = timeout_budget or server.timeout_budget

    def allocate(self, request: HandoffAllocateRequest) -> HandoffProtocolResult:
        operation_started_at = self._now()
        allocated = self._server.allocate(request)
        if not allocated.ok or allocated.allocation is None:
            return allocated
        allocation = allocated.allocation
        if self._client_timed_out(operation_started_at):
            self._compensate(allocation)
            return _failure(
                HandoffFailureCode.CLIENT_TIMEOUT,
                "allocation exceeded client_timeout_s",
            )
        persisted = self._router_ledger.persist(allocation)
        if not persisted.ok:
            self._compensate(allocation)
            return persisted
        try:
            self._run_fault(HandoffFaultPoint.AFTER_ROUTER_PERSIST, request.identity)
        except HandoffBoundaryFailure as exc:
            self._compensate(allocation)
            return _failure(HandoffFailureCode.BOUNDARY_FAILURE, str(exc))
        except BaseException:
            self._compensate(allocation)
            raise
        if self._client_timed_out(operation_started_at):
            self._compensate(allocation)
            return _failure(
                HandoffFailureCode.CLIENT_TIMEOUT,
                "allocation exceeded client_timeout_s",
            )
        try:
            confirmed = self._server.confirm(
                HandoffConfirmRequest(
                    request.identity,
                    allocation.delivery_token,
                    request.scheduler_epoch,
                )
            )
        except BaseException:
            self._compensate(allocation)
            raise
        if not confirmed.ok or confirmed.allocation is None:
            self._compensate(allocation)
            return confirmed
        if self._client_timed_out(operation_started_at):
            self._compensate(confirmed.allocation)
            return _failure(
                HandoffFailureCode.CLIENT_TIMEOUT,
                "allocation exceeded client_timeout_s",
            )
        recorded = self._router_ledger.confirm(confirmed.allocation)
        if not recorded.ok:
            self._server.lifecycle(
                HandoffLifecycleRequest(
                    request.identity,
                    HandoffLifecycleEvent.ABORT,
                    request.scheduler_epoch,
                )
            )
            self._router_ledger.remove(request.identity)
            return recorded
        return confirmed

    def dispatch(self, identity: HandoffAttemptId) -> HandoffProtocolResult:
        return self._router_ledger.mark_dispatched(identity)

    def commit_decode(self, request: HandoffDecodeCommitRequest) -> HandoffProtocolResult:
        try:
            return self._server.commit_decode(request)
        except BaseException:
            self._router_ledger.remove(request.identity)
            raise

    def lifecycle(self, request: HandoffLifecycleRequest) -> HandoffProtocolResult:
        try:
            result = self._server.lifecycle(request)
        except BaseException:
            self._router_ledger.remove(request.identity)
            raise
        if result.ok and request.event in (
            HandoffLifecycleEvent.REQUEST_COMPLETE,
            HandoffLifecycleEvent.ABORT,
        ):
            self._router_ledger.remove(request.identity)
        return result

    def revoke(self, request: HandoffRevokeRequest) -> HandoffProtocolResult:
        result = self._server.revoke(request)
        if result.ok:
            self._router_ledger.remove(request.identity)
        return result

    def status(self, request: HandoffStatusRequest) -> HandoffProtocolResult:
        return self._server.status(request)

    def poll(self) -> HandoffServerState:
        state = self._server.poll()
        terminal_states = {
            HandoffAttemptState.REQUEST_COMPLETED,
            HandoffAttemptState.ALLOCATION_REVOKED,
            HandoffAttemptState.ATTEMPT_ABORTED,
        }
        for entry in self._router_ledger.snapshot().entries:
            status = self._server.status(HandoffStatusRequest(entry.identity, entry.identity.scheduler_epoch))
            if not status.ok or (status.attempt is not None and status.attempt.state in terminal_states):
                self._router_ledger.remove(entry.identity)
        return state

    def quiesce(self) -> None:
        self._server.quiesce()
        self._router_ledger.quiesce()

    def _compensate(self, allocation: HandoffAllocation) -> None:
        status = self._server.status(
            HandoffStatusRequest(
                allocation.identity,
                allocation.identity.scheduler_epoch,
            )
        )
        if status.ok and status.attempt is not None and (status.attempt.state is HandoffAttemptState.DELIVERY_PENDING):
            self._server.revoke(
                HandoffRevokeRequest(
                    allocation.identity,
                    allocation.delivery_token,
                    allocation.identity.scheduler_epoch,
                )
            )
        elif (
            status.ok
            and status.attempt is not None
            and status.attempt.state
            not in (
                HandoffAttemptState.REQUEST_COMPLETED,
                HandoffAttemptState.ALLOCATION_REVOKED,
                HandoffAttemptState.ATTEMPT_ABORTED,
            )
        ):
            self._server.lifecycle(
                HandoffLifecycleRequest(
                    allocation.identity,
                    HandoffLifecycleEvent.ABORT,
                    allocation.identity.scheduler_epoch,
                )
            )
        self._router_ledger.remove(allocation.identity)

    def _now(self) -> float:
        return float(self._clock())

    def _client_timed_out(self, started_at: float) -> bool:
        return self._now() - started_at >= self._timeout_budget.client_timeout_s

    def _run_fault(self, point: HandoffFaultPoint, identity: HandoffAttemptId) -> None:
        if self._fault_hook is not None:
            self._fault_hook(point, identity)
