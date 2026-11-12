# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Dormant Trigger and PDHybrid adapters over the TARS single-role protocol."""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from motor.coordinator.scheduler.policy.tars.handoff_protocol import (
    HandoffAllocateRequest,
    HandoffAllocation,
    HandoffAttemptId,
    HandoffAttemptSequencer,
    HandoffDecodeCommitRequest,
    HandoffFailure,
    HandoffFailureCode,
    HandoffLifecycleEvent,
    HandoffLifecycleRequest,
    HandoffProtocolClient,
    HandoffProtocolResult,
    HandoffRole,
)
from motor.coordinator.scheduler.policy.tars.original.models import TarsTarget


class _TriggerState(str, Enum):
    PLACEHOLDER = "placeholder"
    DECODE_BOUND = "decode_bound"
    PREFILL_COMPLETED = "prefill_completed"
    TERMINAL = "terminal"


class _PDHybridState(str, Enum):
    PLACEHOLDER = "placeholder"
    ACTIVE = "active"
    TERMINAL = "terminal"


@dataclass
class _TriggerAttempt:
    identity: HandoffAttemptId
    state: _TriggerState = _TriggerState.PLACEHOLDER
    decode_target: TarsTarget | None = None
    decode_engine_type: str = ""
    topology_epoch: int = -1
    allocation: HandoffAllocation | None = None


@dataclass
class _PDHybridAttempt:
    identity: HandoffAttemptId
    state: _PDHybridState = _PDHybridState.PLACEHOLDER
    allocation: HandoffAllocation | None = None
    terminal_event: HandoffLifecycleEvent | None = None


def _failure(code: HandoffFailureCode, message: str) -> HandoffProtocolResult:
    return HandoffProtocolResult(failure=HandoffFailure(code, message))


class TriggerTarsAdapter:
    """Preserve D-first Trigger timing while applying TARS only in the P callback."""

    def __init__(
        self,
        client: HandoffProtocolClient,
        sequencer: HandoffAttemptSequencer,
    ) -> None:
        self._client = client
        self._sequencer = sequencer
        self._lock = threading.RLock()
        self._current: _TriggerAttempt | None = None
        self._prefill_admission_count = 0
        self._prefill_dispatch_count = 0

    def reserve_attempt(self) -> HandoffAttemptId:
        """Create an empty identity before the existing Decode allocation."""
        with self._lock:
            if self._current is not None and self._current.state is not _TriggerState.TERMINAL:
                raise RuntimeError("the current Trigger attempt is still active")
            identity = self._sequencer.reserve()
            self._current = _TriggerAttempt(identity)
            return identity

    def bind_decode(
        self,
        identity: HandoffAttemptId,
        target: TarsTarget,
        engine_type: str,
        topology_epoch: int,
    ) -> HandoffProtocolResult:
        """Record master's completed D-first allocation without TARS mutation."""
        with self._lock:
            current = self._active(identity)
            if isinstance(current, HandoffProtocolResult):
                return current
            if current.state is not _TriggerState.PLACEHOLDER:
                if (
                    current.decode_target == target
                    and current.decode_engine_type == engine_type
                    and current.topology_epoch == topology_epoch
                ):
                    return HandoffProtocolResult()
                return _failure(HandoffFailureCode.ATTEMPT_CONFLICT, "Trigger Decode binding changed")
            if not engine_type:
                return _failure(HandoffFailureCode.ENGINE_TYPE_MISMATCH, "Decode engine type is empty")
            if topology_epoch < 0:
                return _failure(HandoffFailureCode.STALE_TOPOLOGY, "Decode topology epoch is invalid")
            current.decode_target = target
            current.decode_engine_type = engine_type
            current.topology_epoch = topology_epoch
            current.state = _TriggerState.DECODE_BOUND
            return HandoffProtocolResult()

    def handle_prefill_callback(
        self,
        identity: HandoffAttemptId,
        request: HandoffAllocateRequest,
        dispatch: Callable[[HandoffAllocation], None],
    ) -> HandoffProtocolResult:
        """Allocate, confirm, and dispatch P under one serialized callback boundary."""
        with self._lock:
            current = self._active(identity)
            if isinstance(current, HandoffProtocolResult):
                return current
            if current.state is _TriggerState.PREFILL_COMPLETED:
                return HandoffProtocolResult(allocation=current.allocation)
            if current.state is not _TriggerState.DECODE_BOUND:
                return _failure(HandoffFailureCode.INVALID_STATE, "Trigger Decode is not bound")
            if request.identity != identity or request.role is not HandoffRole.PREFILL:
                return _failure(HandoffFailureCode.ATTEMPT_CONFLICT, "callback allocation identity or role changed")
            if request.expected_topology_epoch != current.topology_epoch:
                return _failure(HandoffFailureCode.STALE_TOPOLOGY, "Trigger P and D topology epochs differ")
            if current.decode_target is None:
                return _failure(HandoffFailureCode.INVALID_STATE, "Trigger Decode target is missing")

            allocated = self._client.allocate(request)
            if not allocated.ok or allocated.allocation is None:
                current.state = _TriggerState.TERMINAL
                return allocated
            current.allocation = allocated.allocation
            self._prefill_admission_count += 1
            if allocated.allocation.engine_type != current.decode_engine_type:
                self._abort_protocol(identity)
                current.state = _TriggerState.TERMINAL
                return _failure(HandoffFailureCode.ENGINE_TYPE_MISMATCH, "Trigger P and D engine types differ")

            dispatched = self._client.dispatch(identity)
            if not dispatched.ok:
                self._abort_protocol(identity)
                current.state = _TriggerState.TERMINAL
                return dispatched
            try:
                dispatch(allocated.allocation)
            except BaseException:
                # asyncio.CancelledError is a BaseException on supported Python versions.
                self._abort_protocol(identity)
                current.state = _TriggerState.TERMINAL
                raise
            self._prefill_dispatch_count += 1

            settled = self._client.lifecycle(
                HandoffLifecycleRequest(
                    identity,
                    HandoffLifecycleEvent.PREFILL_COMPLETE,
                    identity.scheduler_epoch,
                )
            )
            if not settled.ok:
                self._abort_protocol(identity)
                current.state = _TriggerState.TERMINAL
                return settled
            bound = self._client.commit_decode(
                HandoffDecodeCommitRequest(
                    identity,
                    current.decode_target,
                    identity.scheduler_epoch,
                    current.topology_epoch,
                )
            )
            if not bound.ok:
                self._abort_protocol(identity)
                current.state = _TriggerState.TERMINAL
                return bound
            current.state = _TriggerState.PREFILL_COMPLETED
            return allocated

    def decode_terminal(
        self,
        identity: HandoffAttemptId,
        *,
        aborted: bool,
    ) -> HandoffProtocolResult:
        """Apply D terminal only to the matching live attempt."""
        with self._lock:
            current = self._active(identity)
            if isinstance(current, HandoffProtocolResult):
                return current
            if current.allocation is None:
                current.state = _TriggerState.TERMINAL
                return HandoffProtocolResult()
            event = HandoffLifecycleEvent.ABORT if aborted else HandoffLifecycleEvent.REQUEST_COMPLETE
            result = self._client.lifecycle(HandoffLifecycleRequest(identity, event, identity.scheduler_epoch))
            if result.ok:
                current.state = _TriggerState.TERMINAL
            return result

    def quiesce(self) -> None:
        with self._lock:
            if self._current is not None and self._current.state is not _TriggerState.TERMINAL:
                if self._current.allocation is not None:
                    self._abort_protocol(self._current.identity)
                self._current.state = _TriggerState.TERMINAL
            self._client.quiesce()

    @property
    def prefill_admission_count(self) -> int:
        with self._lock:
            return self._prefill_admission_count

    @property
    def prefill_dispatch_count(self) -> int:
        with self._lock:
            return self._prefill_dispatch_count

    def _active(self, identity: HandoffAttemptId) -> _TriggerAttempt | HandoffProtocolResult:
        if self._current is None or self._current.identity != identity or self._current.state is _TriggerState.TERMINAL:
            return _failure(HandoffFailureCode.STALE_ATTEMPT, "Trigger attempt is no longer active")
        return self._current

    def _abort_protocol(self, identity: HandoffAttemptId) -> None:
        self._client.lifecycle(
            HandoffLifecycleRequest(
                identity,
                HandoffLifecycleEvent.ABORT,
                identity.scheduler_epoch,
            )
        )


class PDHybridTarsAdapter:
    """Apply atomic single-role admission to ROLE_U without Prefill approximation."""

    def __init__(
        self,
        client: HandoffProtocolClient,
        sequencer: HandoffAttemptSequencer,
    ) -> None:
        self._client = client
        self._sequencer = sequencer
        self._lock = threading.RLock()
        self._current: _PDHybridAttempt | None = None
        self._admission_count = 0

    def reserve_attempt(self) -> HandoffAttemptId:
        with self._lock:
            if self._current is not None and self._current.state is not _PDHybridState.TERMINAL:
                raise RuntimeError("the current PDHybrid attempt is still active")
            identity = self._sequencer.reserve()
            self._current = _PDHybridAttempt(identity)
            return identity

    def allocate(
        self,
        identity: HandoffAttemptId,
        request: HandoffAllocateRequest,
    ) -> HandoffProtocolResult:
        with self._lock:
            current = self._active(identity)
            if isinstance(current, HandoffProtocolResult):
                return current
            if request.identity != identity or request.role is not HandoffRole.UNIFIED:
                return _failure(HandoffFailureCode.ATTEMPT_CONFLICT, "PDHybrid identity or role changed")
            allocated = self._client.allocate(request)
            if not allocated.ok or allocated.allocation is None:
                current.state = _PDHybridState.TERMINAL
                return allocated
            first_admission = current.allocation is None
            current.allocation = allocated.allocation
            dispatched = self._client.dispatch(identity)
            if not dispatched.ok:
                self._client.lifecycle(
                    HandoffLifecycleRequest(
                        identity,
                        HandoffLifecycleEvent.ABORT,
                        identity.scheduler_epoch,
                    )
                )
                current.state = _PDHybridState.TERMINAL
                return dispatched
            if first_admission:
                self._admission_count += 1
            current.state = _PDHybridState.ACTIVE
            return allocated

    def lifecycle(self, request: HandoffLifecycleRequest) -> HandoffProtocolResult:
        with self._lock:
            current = self._current
            if current is None or current.identity != request.identity:
                return _failure(HandoffFailureCode.STALE_ATTEMPT, "PDHybrid lifecycle is stale")
            if request.scheduler_epoch != request.identity.scheduler_epoch:
                return _failure(HandoffFailureCode.STALE_SCHEDULER_EPOCH, "PDHybrid lifecycle epoch is stale")
            if current.state is _PDHybridState.TERMINAL:
                if current.terminal_event is request.event:
                    return HandoffProtocolResult()
                return _failure(HandoffFailureCode.STALE_ATTEMPT, "PDHybrid lifecycle is terminal")
            if request.event is HandoffLifecycleEvent.PREFILL_COMPLETE:
                return _failure(
                    HandoffFailureCode.INVALID_STATE,
                    "ROLE_U has no physical Prefill completion boundary",
                )
            if current.state is not _PDHybridState.ACTIVE:
                if request.event is HandoffLifecycleEvent.ABORT:
                    current.state = _PDHybridState.TERMINAL
                    current.terminal_event = request.event
                    return HandoffProtocolResult()
                return _failure(HandoffFailureCode.INVALID_STATE, "PDHybrid attempt is not active")
            result = self._client.lifecycle(request)
            if result.ok:
                current.state = _PDHybridState.TERMINAL
                current.terminal_event = request.event
            return result

    def quiesce(self) -> None:
        with self._lock:
            if self._current is not None and self._current.state is not _PDHybridState.TERMINAL:
                if self._current.allocation is not None:
                    self._client.lifecycle(
                        HandoffLifecycleRequest(
                            self._current.identity,
                            HandoffLifecycleEvent.ABORT,
                            self._current.identity.scheduler_epoch,
                        )
                    )
                self._current.state = _PDHybridState.TERMINAL
            self._client.quiesce()

    @property
    def admission_count(self) -> int:
        with self._lock:
            return self._admission_count

    @property
    def physical_prefill_complete_count(self) -> int:
        return 0

    def _active(self, identity: HandoffAttemptId) -> _PDHybridAttempt | HandoffProtocolResult:
        if (
            self._current is None
            or self._current.identity != identity
            or self._current.state is _PDHybridState.TERMINAL
        ):
            return _failure(HandoffFailureCode.STALE_ATTEMPT, "PDHybrid attempt is no longer active")
        return self._current
