# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Dormant atomic P/D admission for native Bootstrap coordination.

The protocol deliberately has no transport or production Router wiring.  It
models one joint Scheduler/Router ownership boundary: Prefill is selected by
TARS, Decode is selected through an injected side-effect-free callback, and
neither target becomes dispatch-visible until both workload ledgers are
committed and one delivery token is confirmed.
"""

import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from motor.coordinator.scheduler.policy.tars.models import TarsCacheEvidence
from motor.coordinator.scheduler.policy.tars.original.models import (
    TarsRequestContext,
    TarsTarget,
)
from motor.coordinator.scheduler.policy.tars.state_runtime import (
    TarsAttemptKey,
    TarsAttemptSpec,
    TarsCapacityError,
    TarsStaleTopologyError,
    TarsStateError,
    TarsStateRuntime,
)


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class BootstrapJointState(str, Enum):
    """Joint-attempt states; no state exposes only one leg to dispatch."""

    PD_ADMITTED = "pd_admitted"
    P_TARGET_RESERVED = "p_target_reserved"
    PD_TARGETS_VALIDATED = "pd_targets_validated"
    P_WORKLOAD_COMMITTED = "p_workload_committed"
    D_WORKLOAD_COMMITTED = "d_workload_committed"
    JOINT_DELIVERY_PENDING = "joint_delivery_pending"
    OWNERSHIP_TRANSFERRED = "ownership_transferred"
    PREFILL_SETTLING = "prefill_settling"
    PREFILL_COMPLETED = "prefill_completed"
    REQUEST_COMPLETED = "request_completed"
    JOINT_ALLOCATION_REVOKING = "joint_allocation_revoking"
    JOINT_ALLOCATION_REVOKED = "joint_allocation_revoked"
    PD_WORKLOAD_RELEASING = "pd_workload_releasing"
    ATTEMPT_ABORTED = "attempt_aborted"


class BootstrapJointRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class BootstrapJointEvent(str, Enum):
    PREFILL_COMPLETE = "prefill_complete"
    REQUEST_COMPLETE = "request_complete"
    ABORT = "abort"


class BootstrapJointFailureCode(str, Enum):
    NON_BOOTSTRAP_MODE = "non_bootstrap_mode"
    STALE_SCHEDULER_EPOCH = "stale_scheduler_epoch"
    STALE_TOPOLOGY = "stale_topology"
    ATTEMPT_CONFLICT = "attempt_conflict"
    ATTEMPT_NOT_FOUND = "attempt_not_found"
    INVALID_STATE = "invalid_state"
    TARGET_MISMATCH = "target_mismatch"
    ENGINE_TYPE_MISMATCH = "engine_type_mismatch"
    CIRCUIT_OPEN = "circuit_open"
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    DELIVERY_TOKEN_MISMATCH = "delivery_token_mismatch"
    ROUTER_LEDGER_CONFLICT = "router_ledger_conflict"
    PRECONFIRM_DISPATCH = "preconfirm_dispatch"
    BOUNDARY_FAILURE = "boundary_failure"
    INTERNAL_STATE = "internal_state"


class BootstrapJointFaultPoint(str, Enum):
    AFTER_P_TARGET_RESERVATION = "after_p_target_reservation"
    AFTER_PD_TARGETS_VALIDATED = "after_pd_targets_validated"
    AFTER_P_WORKLOAD_COMMIT = "after_p_workload_commit"
    AFTER_D_WORKLOAD_COMMIT = "after_d_workload_commit"
    AFTER_DELIVERY_PREPARE = "after_delivery_prepare"
    AFTER_ROUTER_PERSIST = "after_router_persist"
    AFTER_CONFIRM = "after_confirm"
    AFTER_PREFILL_RELEASE = "after_prefill_release"
    AFTER_DECODE_RELEASE = "after_decode_release"


class BootstrapJointBoundaryFailure(RuntimeError):
    """Injected or observed failure at a joint-ownership boundary."""


class _BootstrapJointRejected(RuntimeError):
    def __init__(self, code: BootstrapJointFailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, order=True)
class BootstrapAttemptId:
    request_id: str
    attempt_seq: int
    scheduler_epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not isinstance(self.attempt_seq, int) or isinstance(self.attempt_seq, bool) or self.attempt_seq <= 0:
            raise ValueError("attempt_seq must be positive")
        if not _is_nonnegative_int(self.scheduler_epoch):
            raise ValueError("scheduler_epoch must be non-negative")

    @property
    def key(self) -> TarsAttemptKey:
        return TarsAttemptKey(self.request_id, self.attempt_seq)


class BootstrapAttemptSequencer:
    """Reserve a fresh Bootstrap attempt only after preflight succeeds."""

    def __init__(self, root_request_id: str, scheduler_epoch: int) -> None:
        if not root_request_id:
            raise ValueError("root_request_id must be non-empty")
        if not _is_nonnegative_int(scheduler_epoch):
            raise ValueError("scheduler_epoch must be non-negative")
        self._root_request_id = root_request_id
        self._scheduler_epoch = scheduler_epoch
        self._next_attempt_seq = 1
        self._lock = threading.Lock()

    def reserve(self) -> BootstrapAttemptId:
        with self._lock:
            identity = BootstrapAttemptId(
                self._root_request_id,
                self._next_attempt_seq,
                self._scheduler_epoch,
            )
            self._next_attempt_seq += 1
            return identity


@dataclass(frozen=True)
class BootstrapJointTargetSpec:
    target: TarsTarget
    role: BootstrapJointRole
    engine_type: str
    circuit_open: bool = False
    decode_token_capacity: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.engine_type, str) or not self.engine_type.strip():
            raise ValueError("engine_type must be non-empty")
        if not _is_nonnegative_int(self.decode_token_capacity):
            raise ValueError("decode_token_capacity must be non-negative")
        if self.role is BootstrapJointRole.DECODE and self.decode_token_capacity == 0:
            raise ValueError("Decode targets require positive token capacity")


@dataclass(frozen=True)
class BootstrapJointAdmissionSpec:
    """Identity-free input used by the side-effect-free Bootstrap preflight."""

    request_id: str
    session_id: str
    input_tokens: int
    prefill_evidence: tuple[TarsCacheEvidence, ...]
    max_output_tokens: int
    block_size: int
    decode_workload_tokens: int
    scheduler_epoch: int
    expected_topology_epoch: int
    engine_type: str = "sglang"
    coordination_mode: str = "bootstrap"

    def __post_init__(self) -> None:
        if not self.request_id or not self.session_id:
            raise ValueError("request_id and session_id must be non-empty")
        for field_name in ("input_tokens", "max_output_tokens", "scheduler_epoch", "expected_topology_epoch"):
            if not _is_nonnegative_int(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be non-negative")
        if not _is_nonnegative_int(self.block_size) or self.block_size == 0:
            raise ValueError("block_size must be positive")
        if not _is_nonnegative_int(self.decode_workload_tokens) or self.decode_workload_tokens == 0:
            raise ValueError("decode_workload_tokens must be positive")
        if not self.prefill_evidence:
            raise ValueError("prefill_evidence must not be empty")
        targets = tuple(item.target for item in self.prefill_evidence)
        if len(targets) != len(set(targets)):
            raise ValueError("prefill evidence targets must be unique")
        if any(item.input_tokens != self.input_tokens for item in self.prefill_evidence):
            raise ValueError("prefill evidence token counts must match input_tokens")
        if not self.engine_type.strip():
            raise ValueError("engine_type must be non-empty")

    def materialize(self, identity: BootstrapAttemptId) -> TarsAttemptSpec:
        if identity.request_id != self.request_id:
            raise ValueError("attempt identity and admission request differ")
        return TarsAttemptSpec(
            key=identity.key,
            request=TarsRequestContext(
                session_id=self.session_id,
                request_id=self.request_id,
                input_tokens=self.input_tokens,
            ),
            evidence=self.prefill_evidence,
            max_output_tokens=self.max_output_tokens,
            block_size=self.block_size,
        )


@dataclass(frozen=True)
class BootstrapDecodeCandidate:
    target: TarsTarget
    committed_tokens: int
    available_tokens: int


DecodeSelector = Callable[
    [BootstrapJointAdmissionSpec, tuple[BootstrapDecodeCandidate, ...]],
    TarsTarget,
]


@dataclass(frozen=True)
class BootstrapJointAllocateRequest:
    identity: BootstrapAttemptId
    admission: BootstrapJointAdmissionSpec


@dataclass(frozen=True)
class BootstrapJointConfirmRequest:
    identity: BootstrapAttemptId
    delivery_token: str
    scheduler_epoch: int


@dataclass(frozen=True)
class BootstrapJointRevokeRequest:
    identity: BootstrapAttemptId
    delivery_token: str
    scheduler_epoch: int


@dataclass(frozen=True)
class BootstrapJointLifecycleRequest:
    identity: BootstrapAttemptId
    event: BootstrapJointEvent
    scheduler_epoch: int
    observed_ttft: float | None = None


@dataclass(frozen=True)
class BootstrapJointFailure:
    code: BootstrapJointFailureCode
    message: str


@dataclass(frozen=True)
class BootstrapJointAllocation:
    identity: BootstrapAttemptId
    prefill_target: TarsTarget
    decode_target: TarsTarget
    delivery_token: str
    topology_epoch: int
    engine_type: str
    confirmed: bool = False
    dispatch_visible: bool = False


@dataclass(frozen=True)
class BootstrapJointAttemptSnapshot:
    identity: BootstrapAttemptId
    state: BootstrapJointState
    prefill_target: TarsTarget
    decode_target: TarsTarget | None
    topology_epoch: int
    engine_type: str


@dataclass(frozen=True)
class BootstrapJointResult:
    failure: BootstrapJointFailure | None = None
    allocation: BootstrapJointAllocation | None = None
    attempt: BootstrapJointAttemptSnapshot | None = None

    @property
    def ok(self) -> bool:
        return self.failure is None


@dataclass(frozen=True)
class BootstrapJointServerSnapshot:
    scheduler_epoch: int
    active_attempts: tuple[BootstrapJointAttemptSnapshot, ...]
    pending_deliveries: tuple[BootstrapJointAllocation, ...]
    decode_workloads: tuple[tuple[BootstrapAttemptId, TarsTarget, int], ...]
    terminal_receipts: tuple[BootstrapJointAttemptSnapshot, ...]
    target_inventory: tuple[BootstrapJointTargetSpec, ...]


@dataclass(frozen=True)
class BootstrapJointRouterSnapshot:
    entries: tuple[BootstrapJointAllocation, ...]

    @property
    def visible_entries(self) -> tuple[BootstrapJointAllocation, ...]:
        return tuple(item for item in self.entries if item.dispatch_visible)


@dataclass
class _AttemptRecord:
    request: BootstrapJointAllocateRequest
    state: BootstrapJointState
    prefill_target: TarsTarget
    engine_type: str
    decode_target: TarsTarget | None = None
    delivery_token: str = ""
    prefill_released: bool = False


FaultHook = Callable[[BootstrapJointFaultPoint, BootstrapAttemptId], None]


def _success(
    *,
    allocation: BootstrapJointAllocation | None = None,
    attempt: BootstrapJointAttemptSnapshot | None = None,
) -> BootstrapJointResult:
    return BootstrapJointResult(allocation=allocation, attempt=attempt)


def _failure(code: BootstrapJointFailureCode, message: str) -> BootstrapJointResult:
    return BootstrapJointResult(failure=BootstrapJointFailure(code, message))


class BootstrapJointProtocolServer:
    """Scheduler-side owner of one atomic Bootstrap P/D admission."""

    def __init__(
        self,
        runtime: TarsStateRuntime,
        scheduler_epoch: int,
        decode_selector: DecodeSelector,
        fault_hook: FaultHook | None = None,
    ) -> None:
        if not _is_nonnegative_int(scheduler_epoch):
            raise ValueError("scheduler_epoch must be non-negative")
        if not callable(decode_selector):
            raise ValueError("decode_selector must be callable")
        self._runtime = runtime
        self._scheduler_epoch = scheduler_epoch
        self._decode_selector = decode_selector
        self._fault_hook = fault_hook
        self._lock = threading.RLock()
        self._targets: dict[TarsTarget, BootstrapJointTargetSpec] = {}
        self._decode_usage: dict[TarsTarget, int] = {}
        self._decode_workloads: dict[BootstrapAttemptId, tuple[TarsTarget, int]] = {}
        self._attempts: dict[BootstrapAttemptId, _AttemptRecord] = {}
        self._pending_deliveries: dict[BootstrapAttemptId, BootstrapJointAllocation] = {}
        self._terminal_receipts: dict[BootstrapAttemptId, BootstrapJointAttemptSnapshot] = {}
        self._terminal_tokens: dict[BootstrapAttemptId, str] = {}

    def sync_targets(
        self,
        topology_epoch: int,
        targets: tuple[BootstrapJointTargetSpec, ...],
    ) -> bool:
        with self._lock:
            if topology_epoch != self._runtime.topology_epoch:
                return False
            target_ids = tuple(item.target for item in targets)
            if len(target_ids) != len(set(target_ids)):
                raise ValueError("Bootstrap targets must be unique")
            runtime_targets = {item.target for item in self._runtime.snapshot().replicas}
            if set(target_ids) != runtime_targets:
                raise ValueError("Bootstrap target inventory must match TARS topology")
            referenced = {target for target, _tokens in self._decode_workloads.values()}
            if not referenced <= set(target_ids):
                raise ValueError("active Decode workloads prevent target removal")
            self._targets = {item.target: item for item in targets}
            self._decode_usage = {
                item.target: self._decode_usage.get(item.target, 0)
                for item in targets
                if item.role is BootstrapJointRole.DECODE
            }
            self._assert_invariants_locked()
            return True

    def preflight(self, admission: BootstrapJointAdmissionSpec) -> BootstrapJointResult:
        """Validate Bootstrap mode and inventory without reserving identity or state."""
        with self._lock:
            return self._preflight_locked(admission) or _success()

    def allocate(self, request: BootstrapJointAllocateRequest) -> BootstrapJointResult:
        with self._lock:
            invalid = self._validate_allocate_request_locked(request)
            if invalid is not None:
                return invalid
            if request.identity in self._attempts or request.identity in self._terminal_receipts:
                return _failure(
                    BootstrapJointFailureCode.ATTEMPT_CONFLICT,
                    "duplicate joint admission is not supported",
                )
            record = _AttemptRecord(
                request=request,
                state=BootstrapJointState.PD_ADMITTED,
                prefill_target=request.admission.prefill_evidence[0].target,
                engine_type=request.admission.engine_type.strip().lower(),
            )
            self._attempts[request.identity] = record
            try:
                self._allocate_stages_locked(record)
                return _success(
                    allocation=self._pending_deliveries[request.identity],
                    attempt=self._snapshot_attempt(record),
                )
            except BootstrapJointBoundaryFailure as exc:
                self._revoke_locked(record)
                return _failure(BootstrapJointFailureCode.BOUNDARY_FAILURE, str(exc))
            except _BootstrapJointRejected as exc:
                self._revoke_locked(record)
                return _failure(exc.code, str(exc))
            except TarsStaleTopologyError as exc:
                self._revoke_locked(record)
                return _failure(BootstrapJointFailureCode.STALE_TOPOLOGY, str(exc))
            except TarsCapacityError as exc:
                self._revoke_locked(record)
                return _failure(BootstrapJointFailureCode.CAPACITY_EXHAUSTED, str(exc))
            except TarsStateError as exc:
                self._revoke_locked(record)
                return _failure(BootstrapJointFailureCode.INTERNAL_STATE, str(exc))
            except BaseException:
                self._revoke_locked(record)
                raise

    def confirm(self, request: BootstrapJointConfirmRequest) -> BootstrapJointResult:
        with self._lock:
            invalid = self._validate_epoch(request.identity, request.scheduler_epoch)
            if invalid is not None:
                return invalid
            record = self._attempts.get(request.identity)
            if record is None:
                return self._terminal_or_missing(request.identity)
            if request.delivery_token != record.delivery_token:
                return _failure(
                    BootstrapJointFailureCode.DELIVERY_TOKEN_MISMATCH,
                    "joint delivery token does not match",
                )
            if record.state is not BootstrapJointState.JOINT_DELIVERY_PENDING:
                return _failure(
                    BootstrapJointFailureCode.INVALID_STATE,
                    "joint allocation is not awaiting confirmation",
                )
            record.state = BootstrapJointState.OWNERSHIP_TRANSFERRED
            self._pending_deliveries.pop(request.identity, None)
            try:
                self._run_fault(BootstrapJointFaultPoint.AFTER_CONFIRM, request.identity)
            except BootstrapJointBoundaryFailure as exc:
                self._abort_locked(record)
                return _failure(BootstrapJointFailureCode.BOUNDARY_FAILURE, str(exc))
            except BaseException:
                self._abort_locked(record)
                raise
            self._assert_invariants_locked()
            return _success(
                allocation=self._allocation(record, confirmed=True),
                attempt=self._snapshot_attempt(record),
            )

    def revoke(self, request: BootstrapJointRevokeRequest) -> BootstrapJointResult:
        with self._lock:
            invalid = self._validate_epoch(request.identity, request.scheduler_epoch)
            if invalid is not None:
                return invalid
            record = self._attempts.get(request.identity)
            if record is None:
                return self._terminal_or_missing(request.identity)
            if request.delivery_token != record.delivery_token:
                return _failure(
                    BootstrapJointFailureCode.DELIVERY_TOKEN_MISMATCH,
                    "joint revocation token does not match",
                )
            if record.state is not BootstrapJointState.JOINT_DELIVERY_PENDING:
                return _failure(
                    BootstrapJointFailureCode.INVALID_STATE,
                    "confirmed joint ownership cannot be revoked",
                )
            return self._revoke_locked(record)

    def lifecycle(self, request: BootstrapJointLifecycleRequest) -> BootstrapJointResult:
        with self._lock:
            invalid = self._validate_epoch(request.identity, request.scheduler_epoch)
            if invalid is not None:
                return invalid
            terminal = self._terminal_receipts.get(request.identity)
            if terminal is not None:
                if self._terminal_matches_event(terminal.state, request.event):
                    return _success(attempt=terminal)
                return _failure(BootstrapJointFailureCode.INVALID_STATE, "attempt is terminal")
            record = self._attempts.get(request.identity)
            if record is None:
                return _failure(BootstrapJointFailureCode.ATTEMPT_NOT_FOUND, "attempt does not exist")
            if request.event is BootstrapJointEvent.PREFILL_COMPLETE:
                return self._settle_prefill_locked(record, request.observed_ttft)
            if request.event is BootstrapJointEvent.REQUEST_COMPLETE:
                return self._complete_request_locked(record)
            return self._abort_locked(record)

    def status(self, identity: BootstrapAttemptId, scheduler_epoch: int) -> BootstrapJointResult:
        with self._lock:
            invalid = self._validate_epoch(identity, scheduler_epoch)
            if invalid is not None:
                return invalid
            record = self._attempts.get(identity)
            if record is not None:
                allocation = None
                if record.delivery_token:
                    allocation = self._allocation(
                        record,
                        confirmed=record.state is not BootstrapJointState.JOINT_DELIVERY_PENDING,
                    )
                return _success(allocation=allocation, attempt=self._snapshot_attempt(record))
            return self._terminal_or_missing(identity)

    def snapshot(self) -> BootstrapJointServerSnapshot:
        with self._lock:
            return BootstrapJointServerSnapshot(
                scheduler_epoch=self._scheduler_epoch,
                active_attempts=tuple(self._snapshot_attempt(record) for _, record in sorted(self._attempts.items())),
                pending_deliveries=tuple(allocation for _, allocation in sorted(self._pending_deliveries.items())),
                decode_workloads=tuple(
                    (identity, target, tokens) for identity, (target, tokens) in sorted(self._decode_workloads.items())
                ),
                terminal_receipts=tuple(receipt for _, receipt in sorted(self._terminal_receipts.items())),
                target_inventory=tuple(item for _, item in sorted(self._targets.items())),
            )

    def quiesce(self) -> None:
        with self._lock:
            for record in tuple(self._attempts.values()):
                self._abort_locked(record)
            self._pending_deliveries.clear()
            self._decode_workloads.clear()
            for target in self._decode_usage:
                self._decode_usage[target] = 0
            self._terminal_receipts.clear()
            self._terminal_tokens.clear()
            self._runtime.quiesce()
            self._assert_invariants_locked()

    def assert_invariants(self) -> None:
        with self._lock:
            self._assert_invariants_locked()
            self._runtime.assert_invariants()

    def _allocate_stages_locked(self, record: _AttemptRecord) -> None:
        request = record.request
        admission = request.admission
        runtime_attempt = self._runtime.enqueue(admission.materialize(request.identity))
        try:
            runtime_attempt = self._runtime.lock_final_target(
                request.identity.key,
                expected_topology_epoch=admission.expected_topology_epoch,
            )
        except TarsCapacityError:
            self._runtime.rebalance(
                request.identity.key,
                expected_topology_epoch=admission.expected_topology_epoch,
            )
            runtime_attempt = self._runtime.lock_final_target(
                request.identity.key,
                expected_topology_epoch=admission.expected_topology_epoch,
            )
        record.prefill_target = runtime_attempt.final_target or runtime_attempt.selected_target
        record.state = BootstrapJointState.P_TARGET_RESERVED
        self._run_fault(BootstrapJointFaultPoint.AFTER_P_TARGET_RESERVATION, request.identity)

        candidates = self._decode_candidates_locked(admission)
        record.decode_target = self._decode_selector(admission, candidates)
        self._validate_final_targets_locked(record)
        record.state = BootstrapJointState.PD_TARGETS_VALIDATED
        self._run_fault(BootstrapJointFaultPoint.AFTER_PD_TARGETS_VALIDATED, request.identity)

        self._runtime.reserve(
            request.identity.key,
            expected_topology_epoch=admission.expected_topology_epoch,
        )
        record.state = BootstrapJointState.P_WORKLOAD_COMMITTED
        self._run_fault(BootstrapJointFaultPoint.AFTER_P_WORKLOAD_COMMIT, request.identity)

        self._commit_decode_locked(record)
        record.state = BootstrapJointState.D_WORKLOAD_COMMITTED
        self._run_fault(BootstrapJointFaultPoint.AFTER_D_WORKLOAD_COMMIT, request.identity)

        record.delivery_token = self._delivery_token(record)
        record.state = BootstrapJointState.JOINT_DELIVERY_PENDING
        self._pending_deliveries[request.identity] = self._allocation(record, confirmed=False)
        self._run_fault(BootstrapJointFaultPoint.AFTER_DELIVERY_PREPARE, request.identity)
        self._assert_invariants_locked()

    def _preflight_locked(
        self,
        admission: BootstrapJointAdmissionSpec,
    ) -> BootstrapJointResult | None:
        if admission.coordination_mode.strip().lower() != "bootstrap":
            return _failure(
                BootstrapJointFailureCode.NON_BOOTSTRAP_MODE,
                "joint P/D admission is Bootstrap-only",
            )
        if admission.scheduler_epoch != self._scheduler_epoch:
            return _failure(BootstrapJointFailureCode.STALE_SCHEDULER_EPOCH, "Scheduler epoch is stale")
        if admission.expected_topology_epoch != self._runtime.topology_epoch:
            return _failure(BootstrapJointFailureCode.STALE_TOPOLOGY, "topology epoch is stale")
        engine_type = admission.engine_type.strip().lower()
        for evidence in admission.prefill_evidence:
            target = self._targets.get(evidence.target)
            if target is None or target.role is not BootstrapJointRole.PREFILL:
                return _failure(
                    BootstrapJointFailureCode.TARGET_MISMATCH,
                    "Prefill evidence includes a non-Prefill target",
                )
            if target.engine_type.strip().lower() != engine_type:
                return _failure(
                    BootstrapJointFailureCode.ENGINE_TYPE_MISMATCH,
                    "Prefill target engine type differs from admission",
                )
            if target.circuit_open:
                return _failure(
                    BootstrapJointFailureCode.CIRCUIT_OPEN,
                    "Prefill target circuit is open",
                )
        eligible = self._decode_candidates_locked(admission)
        if not eligible:
            decode_targets = tuple(item for item in self._targets.values() if item.role is BootstrapJointRole.DECODE)
            if not any(item.engine_type.strip().lower() == engine_type for item in decode_targets):
                return _failure(
                    BootstrapJointFailureCode.ENGINE_TYPE_MISMATCH,
                    "no Decode target has the required engine type",
                )
            if all(item.circuit_open for item in decode_targets):
                return _failure(
                    BootstrapJointFailureCode.CIRCUIT_OPEN,
                    "all compatible Decode target circuits are open",
                )
            return _failure(
                BootstrapJointFailureCode.CAPACITY_EXHAUSTED,
                "no compatible Decode target has capacity",
            )
        return None

    def _validate_allocate_request_locked(
        self,
        request: BootstrapJointAllocateRequest,
    ) -> BootstrapJointResult | None:
        invalid = self._preflight_locked(request.admission)
        if invalid is not None:
            return invalid
        if request.identity.request_id != request.admission.request_id:
            return _failure(BootstrapJointFailureCode.ATTEMPT_CONFLICT, "attempt request ID differs")
        return self._validate_epoch(request.identity, request.admission.scheduler_epoch)

    def _validate_final_targets_locked(self, record: _AttemptRecord) -> None:
        admission = record.request.admission
        if admission.expected_topology_epoch != self._runtime.topology_epoch:
            raise TarsStaleTopologyError("topology changed before joint workload commit")
        prefill = self._targets.get(record.prefill_target)
        decode = self._targets.get(record.decode_target)
        if prefill is None or prefill.role is not BootstrapJointRole.PREFILL:
            raise _BootstrapJointRejected(
                BootstrapJointFailureCode.TARGET_MISMATCH,
                "selected Prefill target is invalid",
            )
        if decode is None or decode.role is not BootstrapJointRole.DECODE:
            raise _BootstrapJointRejected(
                BootstrapJointFailureCode.TARGET_MISMATCH,
                "Decode selector returned an invalid target",
            )
        if prefill.circuit_open or decode.circuit_open:
            raise _BootstrapJointRejected(
                BootstrapJointFailureCode.CIRCUIT_OPEN,
                "P/D circuit is open",
            )
        required_engine = admission.engine_type.strip().lower()
        if {
            prefill.engine_type.strip().lower(),
            decode.engine_type.strip().lower(),
        } != {required_engine}:
            raise _BootstrapJointRejected(
                BootstrapJointFailureCode.ENGINE_TYPE_MISMATCH,
                "P/D engine types differ",
            )
        available = decode.decode_token_capacity - self._decode_usage.get(decode.target, 0)
        if available < admission.decode_workload_tokens:
            raise TarsCapacityError("Decode target has insufficient capacity")

    def _decode_candidates_locked(
        self,
        admission: BootstrapJointAdmissionSpec,
    ) -> tuple[BootstrapDecodeCandidate, ...]:
        engine_type = admission.engine_type.strip().lower()
        candidates = []
        for target, spec in sorted(self._targets.items()):
            if spec.role is not BootstrapJointRole.DECODE or spec.circuit_open:
                continue
            if spec.engine_type.strip().lower() != engine_type:
                continue
            committed = self._decode_usage.get(target, 0)
            available = spec.decode_token_capacity - committed
            if available >= admission.decode_workload_tokens:
                candidates.append(BootstrapDecodeCandidate(target, committed, available))
        return tuple(candidates)

    def _commit_decode_locked(self, record: _AttemptRecord) -> None:
        assert record.decode_target is not None
        if record.request.admission.expected_topology_epoch != self._runtime.topology_epoch:
            raise TarsStaleTopologyError("topology changed before Decode workload commit")
        tokens = record.request.admission.decode_workload_tokens
        spec = self._targets[record.decode_target]
        committed = self._decode_usage.get(record.decode_target, 0)
        if spec.decode_token_capacity - committed < tokens:
            raise TarsCapacityError("Decode target lost capacity before commit")
        self._decode_usage[record.decode_target] = committed + tokens
        self._decode_workloads[record.request.identity] = (record.decode_target, tokens)

    def _release_decode_locked(self, identity: BootstrapAttemptId) -> bool:
        workload = self._decode_workloads.pop(identity, None)
        if workload is None:
            return False
        target, tokens = workload
        self._decode_usage[target] -= tokens
        return True

    def _settle_prefill_locked(
        self,
        record: _AttemptRecord,
        observed_ttft: float | None,
    ) -> BootstrapJointResult:
        if record.state is BootstrapJointState.PREFILL_COMPLETED:
            return _success(attempt=self._snapshot_attempt(record))
        if record.state is not BootstrapJointState.OWNERSHIP_TRANSFERRED:
            return _failure(
                BootstrapJointFailureCode.INVALID_STATE,
                "Prefill completion requires transferred joint ownership",
            )
        record.state = BootstrapJointState.PREFILL_SETTLING
        if not record.prefill_released:
            if not self._runtime.complete_prefill(
                record.request.identity.key,
                observed_ttft,
                keep_running=False,
            ):
                return _failure(BootstrapJointFailureCode.INTERNAL_STATE, "Prefill workload is missing")
            record.prefill_released = True
        try:
            self._run_fault(BootstrapJointFaultPoint.AFTER_PREFILL_RELEASE, record.request.identity)
        except BootstrapJointBoundaryFailure as exc:
            aborted = self._abort_locked(record)
            return BootstrapJointResult(
                failure=BootstrapJointFailure(BootstrapJointFailureCode.BOUNDARY_FAILURE, str(exc)),
                attempt=aborted.attempt,
            )
        except BaseException:
            self._abort_locked(record)
            raise
        record.state = BootstrapJointState.PREFILL_COMPLETED
        self._assert_invariants_locked()
        return _success(attempt=self._snapshot_attempt(record))

    def _complete_request_locked(self, record: _AttemptRecord) -> BootstrapJointResult:
        if record.state is not BootstrapJointState.PREFILL_COMPLETED:
            return _failure(
                BootstrapJointFailureCode.INVALID_STATE,
                "request completion requires Prefill completion",
            )
        record.state = BootstrapJointState.PD_WORKLOAD_RELEASING
        if not self._release_decode_locked(record.request.identity):
            return _failure(BootstrapJointFailureCode.INTERNAL_STATE, "Decode workload is missing")
        try:
            self._run_fault(BootstrapJointFaultPoint.AFTER_DECODE_RELEASE, record.request.identity)
        except BootstrapJointBoundaryFailure as exc:
            aborted = self._finish_locked(record, BootstrapJointState.ATTEMPT_ABORTED)
            return BootstrapJointResult(
                failure=BootstrapJointFailure(BootstrapJointFailureCode.BOUNDARY_FAILURE, str(exc)),
                attempt=aborted.attempt,
            )
        except BaseException:
            self._finish_locked(record, BootstrapJointState.ATTEMPT_ABORTED)
            raise
        return self._finish_locked(record, BootstrapJointState.REQUEST_COMPLETED)

    def _revoke_locked(self, record: _AttemptRecord) -> BootstrapJointResult:
        record.state = BootstrapJointState.JOINT_ALLOCATION_REVOKING
        self._release_decode_locked(record.request.identity)
        self._runtime.abort(record.request.identity.key)
        self._pending_deliveries.pop(record.request.identity, None)
        return self._finish_locked(record, BootstrapJointState.JOINT_ALLOCATION_REVOKED)

    def _abort_locked(self, record: _AttemptRecord) -> BootstrapJointResult:
        record.state = BootstrapJointState.PD_WORKLOAD_RELEASING
        self._release_decode_locked(record.request.identity)
        self._runtime.abort(record.request.identity.key)
        self._pending_deliveries.pop(record.request.identity, None)
        return self._finish_locked(record, BootstrapJointState.ATTEMPT_ABORTED)

    def _finish_locked(
        self,
        record: _AttemptRecord,
        state: BootstrapJointState,
    ) -> BootstrapJointResult:
        identity = record.request.identity
        record.state = state
        receipt = self._snapshot_attempt(record)
        self._attempts.pop(identity, None)
        self._terminal_receipts[identity] = receipt
        self._terminal_tokens[identity] = record.delivery_token
        self._assert_invariants_locked()
        return _success(attempt=receipt)

    def _validate_epoch(
        self,
        identity: BootstrapAttemptId,
        scheduler_epoch: int,
    ) -> BootstrapJointResult | None:
        if scheduler_epoch != self._scheduler_epoch or identity.scheduler_epoch != self._scheduler_epoch:
            return _failure(BootstrapJointFailureCode.STALE_SCHEDULER_EPOCH, "Scheduler epoch is stale")
        return None

    def _terminal_or_missing(self, identity: BootstrapAttemptId) -> BootstrapJointResult:
        terminal = self._terminal_receipts.get(identity)
        if terminal is None:
            return _failure(BootstrapJointFailureCode.ATTEMPT_NOT_FOUND, "attempt does not exist")
        return _success(attempt=terminal)

    def _allocation(
        self,
        record: _AttemptRecord,
        *,
        confirmed: bool,
    ) -> BootstrapJointAllocation:
        assert record.decode_target is not None
        return BootstrapJointAllocation(
            identity=record.request.identity,
            prefill_target=record.prefill_target,
            decode_target=record.decode_target,
            delivery_token=record.delivery_token,
            topology_epoch=record.request.admission.expected_topology_epoch,
            engine_type=record.engine_type,
            confirmed=confirmed,
        )

    @staticmethod
    def _snapshot_attempt(record: _AttemptRecord) -> BootstrapJointAttemptSnapshot:
        return BootstrapJointAttemptSnapshot(
            identity=record.request.identity,
            state=record.state,
            prefill_target=record.prefill_target,
            decode_target=record.decode_target,
            topology_epoch=record.request.admission.expected_topology_epoch,
            engine_type=record.engine_type,
        )

    @staticmethod
    def _terminal_matches_event(
        state: BootstrapJointState,
        event: BootstrapJointEvent,
    ) -> bool:
        return (event is BootstrapJointEvent.REQUEST_COMPLETE and state is BootstrapJointState.REQUEST_COMPLETED) or (
            event is BootstrapJointEvent.ABORT and state is BootstrapJointState.ATTEMPT_ABORTED
        )

    @staticmethod
    def _delivery_token(record: _AttemptRecord) -> str:
        assert record.decode_target is not None
        identity = record.request.identity
        material = (
            f"{identity.request_id}:{identity.attempt_seq}:{identity.scheduler_epoch}:"
            f"{record.prefill_target.instance_id}:{record.prefill_target.dp_rank}:"
            f"{record.decode_target.instance_id}:{record.decode_target.dp_rank}:"
            f"{record.request.admission.expected_topology_epoch}:bootstrap"
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _run_fault(
        self,
        point: BootstrapJointFaultPoint,
        identity: BootstrapAttemptId,
    ) -> None:
        if self._fault_hook is not None:
            self._fault_hook(point, identity)

    def _assert_invariants_locked(self) -> None:
        if set(self._attempts) & set(self._terminal_receipts):
            raise RuntimeError("joint attempt is both active and terminal")
        if not set(self._pending_deliveries) <= set(self._attempts):
            raise RuntimeError("pending joint delivery has no active attempt")
        if not set(self._decode_workloads) <= set(self._attempts):
            raise RuntimeError("Decode workload has no active joint attempt")
        expected_usage = {target: 0 for target in self._decode_usage}
        for identity, (target, tokens) in self._decode_workloads.items():
            record = self._attempts[identity]
            if record.decode_target != target:
                raise RuntimeError("Decode workload target differs from joint attempt")
            expected_usage[target] += tokens
        if expected_usage != self._decode_usage:
            raise RuntimeError("Decode workload accounting is inconsistent")
        for target, usage in self._decode_usage.items():
            spec = self._targets.get(target)
            if spec is None or usage > spec.decode_token_capacity:
                raise RuntimeError("Decode workload exceeds target capacity")
        runtime_workloads = {workload.key for workload in self._runtime.snapshot().workloads}
        for identity, allocation in self._pending_deliveries.items():
            record = self._attempts[identity]
            if record.state is not BootstrapJointState.JOINT_DELIVERY_PENDING:
                raise RuntimeError("pending delivery and joint state disagree")
            if identity not in self._decode_workloads or identity.key not in runtime_workloads:
                raise RuntimeError("pending joint delivery lacks a P or D workload")
            if allocation.confirmed or allocation.dispatch_visible:
                raise RuntimeError("pending joint delivery became visible")


class BootstrapJointRouterLedger:
    """Router ledger that exposes only a confirmed P/D pair."""

    def __init__(self) -> None:
        self._entries: dict[BootstrapAttemptId, BootstrapJointAllocation] = {}
        self._lock = threading.RLock()

    def persist(self, allocation: BootstrapJointAllocation) -> BootstrapJointResult:
        with self._lock:
            existing = self._entries.get(allocation.identity)
            if existing is not None:
                if existing == allocation:
                    return _success(allocation=existing)
                return _failure(
                    BootstrapJointFailureCode.ROUTER_LEDGER_CONFLICT,
                    "Router joint entry changed",
                )
            if allocation.confirmed or allocation.dispatch_visible:
                return _failure(
                    BootstrapJointFailureCode.ROUTER_LEDGER_CONFLICT,
                    "Router can persist only a pending joint allocation",
                )
            self._entries[allocation.identity] = allocation
            return _success(allocation=allocation)

    def confirm(self, allocation: BootstrapJointAllocation) -> BootstrapJointResult:
        with self._lock:
            existing = self._entries.get(allocation.identity)
            if existing is None:
                return _failure(BootstrapJointFailureCode.ATTEMPT_NOT_FOUND, "Router entry is missing")
            if not allocation.confirmed or not self._same_pair(existing, allocation):
                return _failure(
                    BootstrapJointFailureCode.ROUTER_LEDGER_CONFLICT,
                    "joint confirmation changed P/D targets",
                )
            confirmed = BootstrapJointAllocation(
                identity=existing.identity,
                prefill_target=existing.prefill_target,
                decode_target=existing.decode_target,
                delivery_token=existing.delivery_token,
                topology_epoch=existing.topology_epoch,
                engine_type=existing.engine_type,
                confirmed=True,
            )
            self._entries[allocation.identity] = confirmed
            return _success(allocation=confirmed)

    def mark_dispatched(self, identity: BootstrapAttemptId) -> BootstrapJointResult:
        with self._lock:
            existing = self._entries.get(identity)
            if existing is None:
                return _failure(BootstrapJointFailureCode.ATTEMPT_NOT_FOUND, "Router entry is missing")
            if not existing.confirmed:
                return _failure(
                    BootstrapJointFailureCode.PRECONFIRM_DISPATCH,
                    "joint delivery is not confirmed",
                )
            visible = BootstrapJointAllocation(
                identity=existing.identity,
                prefill_target=existing.prefill_target,
                decode_target=existing.decode_target,
                delivery_token=existing.delivery_token,
                topology_epoch=existing.topology_epoch,
                engine_type=existing.engine_type,
                confirmed=True,
                dispatch_visible=True,
            )
            self._entries[identity] = visible
            return _success(allocation=visible)

    def remove(self, identity: BootstrapAttemptId) -> BootstrapJointResult:
        with self._lock:
            self._entries.pop(identity, None)
            return _success()

    def snapshot(self) -> BootstrapJointRouterSnapshot:
        with self._lock:
            return BootstrapJointRouterSnapshot(entries=tuple(entry for _, entry in sorted(self._entries.items())))

    def quiesce(self) -> None:
        with self._lock:
            self._entries.clear()

    @staticmethod
    def _same_pair(
        left: BootstrapJointAllocation,
        right: BootstrapJointAllocation,
    ) -> bool:
        return (
            left.identity,
            left.prefill_target,
            left.decode_target,
            left.delivery_token,
            left.topology_epoch,
            left.engine_type,
        ) == (
            right.identity,
            right.prefill_target,
            right.decode_target,
            right.delivery_token,
            right.topology_epoch,
            right.engine_type,
        )


class BootstrapJointProtocolClient:
    """Run preflight, identity reservation, joint persistence, and confirmation."""

    def __init__(
        self,
        server: BootstrapJointProtocolServer,
        router_ledger: BootstrapJointRouterLedger,
        fault_hook: FaultHook | None = None,
    ) -> None:
        self._server = server
        self._router_ledger = router_ledger
        self._fault_hook = fault_hook

    def allocate(
        self,
        admission: BootstrapJointAdmissionSpec,
        sequencer: BootstrapAttemptSequencer,
    ) -> BootstrapJointResult:
        preflight = self._server.preflight(admission)
        if not preflight.ok:
            return preflight
        identity = sequencer.reserve()
        try:
            request = BootstrapJointAllocateRequest(identity, admission)
            allocated = self._server.allocate(request)
        except ValueError as exc:
            return _failure(BootstrapJointFailureCode.ATTEMPT_CONFLICT, str(exc))
        if not allocated.ok or allocated.allocation is None:
            return allocated
        allocation = allocated.allocation
        persisted = self._router_ledger.persist(allocation)
        if not persisted.ok:
            self._compensate(allocation)
            return persisted
        try:
            self._run_fault(BootstrapJointFaultPoint.AFTER_ROUTER_PERSIST, identity)
        except BootstrapJointBoundaryFailure as exc:
            self._compensate(allocation)
            return _failure(BootstrapJointFailureCode.BOUNDARY_FAILURE, str(exc))
        except BaseException:
            self._compensate(allocation)
            raise
        try:
            confirmed = self._server.confirm(
                BootstrapJointConfirmRequest(identity, allocation.delivery_token, admission.scheduler_epoch)
            )
        except BaseException:
            self._compensate(allocation)
            raise
        if not confirmed.ok or confirmed.allocation is None:
            self._compensate(allocation)
            return confirmed
        recorded = self._router_ledger.confirm(confirmed.allocation)
        if not recorded.ok:
            self._server.lifecycle(
                BootstrapJointLifecycleRequest(
                    identity,
                    BootstrapJointEvent.ABORT,
                    admission.scheduler_epoch,
                )
            )
            self._router_ledger.remove(identity)
            return recorded
        return confirmed

    def dispatch(self, identity: BootstrapAttemptId) -> BootstrapJointResult:
        return self._router_ledger.mark_dispatched(identity)

    def lifecycle(self, request: BootstrapJointLifecycleRequest) -> BootstrapJointResult:
        try:
            result = self._server.lifecycle(request)
        except BaseException:
            self._router_ledger.remove(request.identity)
            raise
        if result.ok and request.event in (
            BootstrapJointEvent.REQUEST_COMPLETE,
            BootstrapJointEvent.ABORT,
        ):
            self._router_ledger.remove(request.identity)
        elif (
            not result.ok
            and result.attempt is not None
            and (result.attempt.state is BootstrapJointState.ATTEMPT_ABORTED)
        ):
            self._router_ledger.remove(request.identity)
        return result

    def quiesce(self) -> None:
        self._server.quiesce()
        self._router_ledger.quiesce()

    def _compensate(self, allocation: BootstrapJointAllocation) -> None:
        status = self._server.status(
            allocation.identity,
            allocation.identity.scheduler_epoch,
        )
        if status.ok and status.attempt is not None:
            if status.attempt.state is BootstrapJointState.JOINT_DELIVERY_PENDING:
                self._server.revoke(
                    BootstrapJointRevokeRequest(
                        allocation.identity,
                        allocation.delivery_token,
                        allocation.identity.scheduler_epoch,
                    )
                )
            elif status.attempt.state not in (
                BootstrapJointState.REQUEST_COMPLETED,
                BootstrapJointState.JOINT_ALLOCATION_REVOKED,
                BootstrapJointState.ATTEMPT_ABORTED,
            ):
                self._server.lifecycle(
                    BootstrapJointLifecycleRequest(
                        allocation.identity,
                        BootstrapJointEvent.ABORT,
                        allocation.identity.scheduler_epoch,
                    )
                )
        self._router_ledger.remove(allocation.identity)

    def _run_fault(
        self,
        point: BootstrapJointFaultPoint,
        identity: BootstrapAttemptId,
    ) -> None:
        if self._fault_hook is not None:
            self._fault_hook(point, identity)
