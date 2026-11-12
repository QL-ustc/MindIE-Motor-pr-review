# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Scheduler-process activation bridge for the TARS reliability protocols.

The pure/state/protocol layers intentionally know nothing about Motor's
InstanceManager workload ledger or ZMQ transport. This bridge is the single
place that maps logical ``(instance_id, dp_rank)`` targets to current Motor
endpoints, commits the existing workload ledger, and exposes pending delivery
records to the Router transport.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from motor.common.logger import get_logger
from motor.common.resources.endpoint import Endpoint, Workload
from motor.common.resources.instance import Instance, PDRole
from motor.config.coordinator import CoordinatorConfig
from motor.coordinator.api_client.conductor_api_client import (
    TENANT_ID,
    ConductorApiClient,
    conductor_instance_id,
)
from motor.coordinator.api_client.conductor_evidence import (
    ConductorEvidence,
    EvidenceAvailability,
    ExactDpEvidence,
    ExpectedDp,
    evaluate_exact_dp_evidence,
)
from motor.coordinator.scheduler.runtime.tars_workload import TarsWorkloadLedger, TarsWorkloadReceipt
from motor.coordinator.scheduler.runtime.workload_shm import WorkloadSharedMemoryOwner
from motor.coordinator.scheduler.policy.tars.bootstrap_joint_protocol import (
    BootstrapAttemptId,
    BootstrapDecodeCandidate,
    BootstrapJointAdmissionSpec,
    BootstrapJointAllocateRequest,
    BootstrapJointConfirmRequest,
    BootstrapJointEvent,
    BootstrapJointLifecycleRequest,
    BootstrapJointProtocolServer,
    BootstrapJointRevokeRequest,
    BootstrapJointRole,
    BootstrapJointTargetSpec,
)
from motor.coordinator.scheduler.policy.tars.conductor_adapter import (
    adapt_conductor_evidence,
)
from motor.coordinator.scheduler.policy.tars.handoff_protocol import (
    HandoffAllocateRequest,
    HandoffAttemptId,
    HandoffConfirmRequest,
    HandoffDecodeCommitRequest,
    HandoffLifecycleEvent,
    HandoffLifecycleRequest,
    HandoffProtocolServer,
    HandoffRevokeRequest,
    HandoffRole,
    HandoffStatusRequest,
    HandoffTargetSpec,
    HandoffTimeoutBudget,
)
from motor.coordinator.scheduler.policy.tars.models import (
    TarsCacheEvidence,
    TarsEvidenceRequest,
)
from motor.coordinator.scheduler.policy.tars.original.models import (
    TarsOriginalConfig,
    TarsRequestContext,
    TarsTarget,
)
from motor.coordinator.scheduler.policy.tars.state_runtime import (
    TarsAttemptSpec,
    TarsReplicaSpec,
)

CircuitOpen = Callable[[int], bool]

logger = get_logger(__name__)


@dataclass
class _ActualCommit:
    target: TarsTarget
    role: PDRole
    workload: Workload
    receipt: TarsWorkloadReceipt


@dataclass
class _PendingActual:
    commits: list[_ActualCommit]
    confirmed: bool
    expires_at: float
    delivery_token: str
    request_fingerprint: str
    response: dict | None = None
    releasing: bool = False


class TarsRuntimeActivation:
    """Own active TARS protocol state inside the authoritative Scheduler process."""

    _SINGLE_MODES = frozenset({"handoff", "trigger", "unified"})

    def __init__(
        self,
        scheduler,
        instance_provider,
        config: CoordinatorConfig,
        *,
        workload_owner: WorkloadSharedMemoryOwner | None,
        circuit_open: CircuitOpen,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._instance_provider = instance_provider
        self._config = config
        self._workload_ledger = TarsWorkloadLedger(workload_owner)
        self._circuit_open = circuit_open
        self._clock = clock or time.monotonic
        tars = config.scheduler_config.tars
        original = TarsOriginalConfig(
            balance_type=tars.balance_type,
            prefill_tpot=tars.prefill_tpot,
            tpct=tars.tpct,
            tprt=tars.tprt,
            ttft_slo=tars.ttft_slo,
            dh_first_balance_ttft_threshold=tars.dh_first_balance_ttft_threshold,
        )
        self._runtime = scheduler.create_tars_state_runtime(
            scheduler_id="scheduler-process",
            config=original,
            clock=self._clock,
        )
        self._scheduler_epoch = time.time_ns()
        timeout_budget = HandoffTimeoutBudget(
            queue_wait_s=tars.queue_wait_s,
            final_query_s=tars.final_query_s,
            exact_commit_s=tars.exact_commit_s,
            settlement_margin_s=tars.settlement_margin_s,
            server_timeout_s=tars.server_timeout_s,
            client_timeout_s=tars.client_timeout_s,
            delivery_lease_s=tars.delivery_lease_s,
            drain_timeout_s=tars.drain_timeout_s,
            recovery_cooldown_s=tars.recovery_cooldown_s,
        )
        self._single = HandoffProtocolServer(
            self._runtime,
            self._scheduler_epoch,
            clock=self._clock,
            timeout_budget=timeout_budget,
        )
        self._joint = BootstrapJointProtocolServer(
            self._runtime,
            self._scheduler_epoch,
            self._select_decode,
        )
        self._topology_fingerprint: tuple | None = None
        self._topology_epoch = -1
        self._target_map: dict[TarsTarget, tuple[Instance, Endpoint]] = {}
        self._actual: dict[tuple[str, int, str], _PendingActual] = {}
        self._settled: OrderedDict[tuple[str, int, str], _PendingActual] = OrderedDict()
        self._draining = False
        self._drain_deadline: float | None = None
        ConductorApiClient.coordinator_config = config

    @property
    def scheduler_epoch(self) -> int:
        return self._scheduler_epoch

    async def prepare_single(self, data: dict) -> dict:
        """Prepare one P/U allocation and return it as delivery-pending."""
        self.poll()
        replay = self._replay_prepared(data, "single")
        if replay is not None:
            return replay
        if self._draining:
            return self._error("server_draining", "TARS Scheduler is draining")
        mode = str(data.get("coordination_mode") or "").strip().lower()
        role = PDRole(str(data.get("role") or ""))
        if mode not in self._SINGLE_MODES:
            return self._error("invalid_mode", f"unsupported single-role mode: {mode}")
        if (mode == "unified") != (role is PDRole.ROLE_U):
            return self._error(
                "invalid_mode",
                "unified mode requires ROLE_U and P/D modes require ROLE_P",
            )

        topology_error = self._sync_topology()
        if topology_error is not None:
            return topology_error
        instances = self._eligible_instances(role, str(data.get("required_engine_type") or ""))
        evidence = await asyncio.to_thread(self._build_evidence, instances, data)
        self.poll()
        replay = self._replay_prepared(data, "single")
        if replay is not None:
            return replay
        if self._draining:
            return self._error("server_draining", "TARS Scheduler is draining")
        if evidence is None:
            self._decision("fallback", data, reason="evidence_unavailable")
            return {"fallback": self._config.scheduler_config.tars.fallback_policy}

        identity = HandoffAttemptId(
            request_id=self._required_text(data, "req_id"),
            attempt_seq=self._positive_int(data, "attempt_seq"),
            scheduler_epoch=self._scheduler_epoch,
        )
        role_value = HandoffRole.UNIFIED if role is PDRole.ROLE_U else HandoffRole.PREFILL
        attempt_spec = self._attempt_spec(identity.request_id, identity.attempt_seq, data, evidence)
        result = self._single.allocate(
            HandoffAllocateRequest(
                identity=identity,
                attempt_spec=attempt_spec,
                scheduler_epoch=self._scheduler_epoch,
                expected_topology_epoch=self._topology_epoch,
                role=role_value,
            )
        )
        if not result.ok or result.allocation is None:
            return self._protocol_error(result.failure)
        allocation = result.allocation
        key = self._key(identity.request_id, identity.attempt_seq, "single")
        pending = self._actual.get(key)
        if pending is not None:
            return pending.response or self._error("workload_releasing", "Workload compensation is pending")
        selected = evidence_by_target(evidence)[allocation.target]
        workload = Workload(active_tokens=float(selected.estimated_compute_tokens))
        commit = self._commit_actual(allocation.target, role, workload)
        if commit is None:
            self._single.revoke(HandoffRevokeRequest(identity, allocation.delivery_token, self._scheduler_epoch))
            return self._error("workload_commit_failed", "Motor workload commit failed")
        self._actual[key] = _PendingActual(
            commits=[commit],
            confirmed=False,
            expires_at=float(allocation.delivery_expires_at or (self._clock() + self._lease_s)),
            delivery_token=allocation.delivery_token,
            request_fingerprint=self._request_fingerprint(data),
        )
        instance, endpoint = self._target_map[allocation.target]
        self._decision("prepared", data, kind="single", target=allocation.target.key)
        response = {
            "kind": "single",
            "identity": self._identity_dict(identity),
            "delivery_token": allocation.delivery_token,
            "instance": instance.model_dump(mode="json"),
            "endpoint": endpoint.model_dump(mode="json"),
            "workload": workload.model_dump(mode="json"),
            "topology_epoch": self._topology_epoch,
        }
        self._actual[key].response = response
        return response

    async def prepare_joint(self, data: dict) -> dict:
        """Prepare a Bootstrap-only P/D pair and return one pending token."""
        self.poll()
        replay = self._replay_prepared(data, "joint")
        if replay is not None:
            return replay
        if self._draining:
            return self._error("server_draining", "TARS Scheduler is draining")
        topology_error = self._sync_topology()
        if topology_error is not None:
            return topology_error
        engine_type = str(data.get("required_engine_type") or "sglang").strip().lower()
        prefill_instances = self._eligible_instances(PDRole.ROLE_P, engine_type)
        evidence = await asyncio.to_thread(self._build_evidence, prefill_instances, data)
        # Evidence I/O yields to shutdown. Joint protocol itself has no drain gate.
        self.poll()
        replay = self._replay_prepared(data, "joint")
        if replay is not None:
            return replay
        if self._draining:
            return self._error("server_draining", "TARS Scheduler is draining")
        if evidence is None:
            self._decision("fallback", data, kind="joint", reason="evidence_unavailable")
            return {"fallback": self._config.scheduler_config.tars.fallback_policy}

        request_id = self._required_text(data, "req_id")
        attempt_seq = self._positive_int(data, "attempt_seq")
        identity = BootstrapAttemptId(request_id, attempt_seq, self._scheduler_epoch)
        input_tokens = len(self._token_ids(data))
        admission = BootstrapJointAdmissionSpec(
            request_id=request_id,
            session_id=self._session_id(data, request_id),
            input_tokens=input_tokens,
            prefill_evidence=evidence,
            max_output_tokens=self._nonnegative_int(data, "max_output_tokens"),
            block_size=self._block_size,
            decode_workload_tokens=self._positive_int(data, "decode_workload_tokens"),
            scheduler_epoch=self._scheduler_epoch,
            expected_topology_epoch=self._topology_epoch,
            engine_type=engine_type,
            coordination_mode="bootstrap",
        )
        preflight = self._joint.preflight(admission)
        if not preflight.ok:
            return self._protocol_error(preflight.failure)
        result = self._joint.allocate(BootstrapJointAllocateRequest(identity, admission))
        if not result.ok or result.allocation is None:
            return self._protocol_error(result.failure)
        allocation = result.allocation
        key = self._key(request_id, attempt_seq, "joint")
        pending = self._actual.get(key)
        if pending is not None:
            return pending.response or self._error("workload_releasing", "Workload compensation is pending")
        selected = evidence_by_target(evidence)[allocation.prefill_target]
        p_workload = Workload(active_tokens=float(selected.estimated_compute_tokens))
        d_workload = Workload(active_tokens=float(admission.decode_workload_tokens))
        commits: list[_ActualCommit] = []
        p_commit = self._commit_actual(allocation.prefill_target, PDRole.ROLE_P, p_workload)
        if p_commit is not None:
            commits.append(p_commit)
        d_commit = None
        if p_commit is not None:
            d_commit = self._commit_actual(allocation.decode_target, PDRole.ROLE_D, d_workload)
            if d_commit is not None:
                commits.append(d_commit)
        self._actual[key] = _PendingActual(
            commits=commits,
            confirmed=False,
            expires_at=self._clock() + self._lease_s,
            delivery_token=allocation.delivery_token,
            request_fingerprint=self._request_fingerprint(data),
        )
        if p_commit is None or d_commit is None:
            self._joint.revoke(BootstrapJointRevokeRequest(identity, allocation.delivery_token, self._scheduler_epoch))
            self._release_pending_actual(request_id, attempt_seq, "joint")
            return self._error("workload_commit_failed", "Motor joint workload commit failed")
        p_instance, p_endpoint = self._target_map[allocation.prefill_target]
        d_instance, d_endpoint = self._target_map[allocation.decode_target]
        self._decision(
            "prepared",
            data,
            kind="joint",
            prefill_target=allocation.prefill_target.key,
            decode_target=allocation.decode_target.key,
        )
        response = {
            "kind": "joint",
            "identity": self._identity_dict(identity),
            "delivery_token": allocation.delivery_token,
            "prefill": self._resource_dict(p_instance, p_endpoint, p_workload),
            "decode": self._resource_dict(d_instance, d_endpoint, d_workload),
            "topology_epoch": self._topology_epoch,
        }
        self._actual[key].response = response
        return response

    def confirm(self, data: dict) -> dict:
        kind, identity, token = self._delivery_input(data)
        if kind == "joint":
            protocol_identity = BootstrapAttemptId(*identity)
            result = self._joint.confirm(BootstrapJointConfirmRequest(protocol_identity, token, self._scheduler_epoch))
        else:
            protocol_identity = HandoffAttemptId(*identity)
            result = self._single.confirm(HandoffConfirmRequest(protocol_identity, token, self._scheduler_epoch))
        if not result.ok:
            return self._protocol_error(result.failure)
        pending = self._actual.get(self._key(identity[0], identity[1], kind))
        if pending is None:
            return self._error("actual_commit_missing", "Motor workload commit is missing")
        pending.confirmed = True
        self._decision("confirmed", data, kind=kind)
        return {"success": True}

    def revoke(self, data: dict) -> dict:
        kind, identity, token = self._delivery_input(data)
        if kind == "joint":
            result = self._joint.revoke(
                BootstrapJointRevokeRequest(BootstrapAttemptId(*identity), token, self._scheduler_epoch)
            )
        else:
            result = self._single.revoke(
                HandoffRevokeRequest(HandoffAttemptId(*identity), token, self._scheduler_epoch)
            )
        if not result.ok:
            return self._protocol_error(result.failure)
        if not self._release_pending_actual(identity[0], identity[1], kind):
            return self._error("workload_release_failed", "Motor workload compensation is pending")
        self._decision("revoked", data, kind=kind)
        return {"success": True}

    def status(self, data: dict) -> dict:
        """Reconcile an ambiguous delivery confirmation without mutating ownership."""
        kind, identity, token = self._delivery_input(data)
        if kind == "joint":
            result = self._joint.status(BootstrapAttemptId(*identity), self._scheduler_epoch)
        else:
            result = self._single.status(HandoffStatusRequest(HandoffAttemptId(*identity), self._scheduler_epoch))
        if not result.ok:
            return self._protocol_error(result.failure)
        allocation = result.allocation
        if allocation is not None and allocation.delivery_token != token:
            return self._error("delivery_token_mismatch", "delivery status token does not match")
        confirmed = bool(allocation is not None and allocation.confirmed)
        pending = self._actual.get(self._key(identity[0], identity[1], kind))
        if pending is not None and confirmed:
            pending.confirmed = True
        attempt = result.attempt
        return {
            "success": True,
            "confirmed": confirmed,
            "state": attempt.state.value if attempt is not None else "unknown",
        }

    def lifecycle(self, data: dict) -> dict:
        kind = str(data.get("kind") or "")
        request_id = self._required_text(data, "req_id")
        attempt_seq = self._positive_int(data, "attempt_seq")
        event = str(data.get("event") or "")
        epoch = self._positive_or_zero_int(data, "scheduler_epoch")
        if kind == "joint":
            result = self._joint.lifecycle(
                BootstrapJointLifecycleRequest(
                    BootstrapAttemptId(request_id, attempt_seq, epoch),
                    BootstrapJointEvent(event),
                    epoch,
                    data.get("observed_ttft"),
                )
            )
        else:
            result = self._single.lifecycle(
                HandoffLifecycleRequest(
                    HandoffAttemptId(request_id, attempt_seq, epoch),
                    HandoffLifecycleEvent(event),
                    epoch,
                    data.get("observed_ttft"),
                )
            )
        if not result.ok:
            return self._protocol_error(result.failure)
        if event in ("request_complete", "abort"):
            if not self._release_pending_actual(request_id, attempt_seq, kind):
                return self._error("workload_release_failed", "Motor workload compensation is pending")
        self._decision("lifecycle", data, kind=kind, lifecycle_event=event)
        return {"success": True}

    def release_workload(self, data: dict) -> dict:
        """Release only a recorded allocation; retries never debit another request."""
        kind, identity, token = self._delivery_input(data)
        if identity[2] != self._scheduler_epoch:
            return self._error("scheduler_epoch_mismatch", "Workload receipt belongs to another Scheduler")
        key = self._key(identity[0], identity[1], kind)
        pending = self._actual.get(key) or self._settled.get(key)
        if pending is None or pending.delivery_token != token:
            return self._error("workload_receipt_missing", "No matching workload receipt")
        instance_id = self._positive_or_zero_int(data, "instance_id")
        endpoint_id = self._positive_or_zero_int(data, "endpoint_id")
        role = str(data.get("role") or "")
        for commit in pending.commits:
            receipt = commit.receipt
            if (receipt.instance_id, receipt.endpoint_id, receipt.role.value) == (instance_id, endpoint_id, role):
                if self._workload_ledger.release(receipt):
                    return {"success": True}
                return self._error("workload_release_failed", "Motor workload release must be retried")
        return self._error("workload_receipt_mismatch", "Target is not owned by this delivery")

    def commit_decode(self, data: dict) -> dict:
        request_id = self._required_text(data, "req_id")
        attempt_seq = self._positive_int(data, "attempt_seq")
        epoch = self._positive_or_zero_int(data, "scheduler_epoch")
        target = TarsTarget(
            str(self._positive_or_zero_int(data, "instance_id")),
            self._positive_or_zero_int(data, "dp_rank"),
        )
        result = self._single.commit_decode(
            HandoffDecodeCommitRequest(
                HandoffAttemptId(request_id, attempt_seq, epoch),
                target,
                epoch,
                self._topology_epoch,
            )
        )
        return {"success": True} if result.ok else self._protocol_error(result.failure)

    def poll(self) -> None:
        """Expire unconfirmed deliveries and reverse their Motor workload commits."""
        self._single.poll()
        now = self._clock()
        for key, pending in tuple(self._actual.items()):
            if not pending.releasing and (pending.confirmed or now < pending.expires_at):
                continue
            request_id, attempt_seq, kind = key
            if kind == "joint":
                snapshot = self._joint.snapshot()
                allocation = next(
                    (
                        item
                        for item in snapshot.pending_deliveries
                        if item.identity.request_id == request_id and item.identity.attempt_seq == attempt_seq
                    ),
                    None,
                )
                if allocation is not None:
                    self._joint.revoke(
                        BootstrapJointRevokeRequest(
                            allocation.identity,
                            allocation.delivery_token,
                            self._scheduler_epoch,
                        )
                    )
            self._release_pending_actual(request_id, attempt_seq, kind)
        if self._draining and self._drain_deadline is not None and now >= self._drain_deadline:
            for request_id, attempt_seq, kind in tuple(self._actual):
                self._release_pending_actual(request_id, attempt_seq, kind)
            self._joint.quiesce()

    def begin_drain(self) -> None:
        if self._draining:
            return
        timeout_s = self._config.scheduler_config.tars.drain_timeout_s
        self._draining = True
        self._drain_deadline = self._clock() + timeout_s
        self._single.begin_drain(timeout_s)

    def drain_complete(self) -> bool:
        """Return true after active single-role and joint ownership has settled."""
        self.poll()
        return (
            not self._actual
            and not self._single.snapshot().active_attempts
            and not self._joint.snapshot().active_attempts
        )

    def quiesce(self) -> None:
        for request_id, attempt_seq, kind in tuple(self._actual):
            self._release_pending_actual(request_id, attempt_seq, kind)
        self._single.quiesce()
        self._joint.quiesce()

    @property
    def _block_size(self) -> int:
        return self._config.scheduler_config.kv_conductor_config.block_size

    @property
    def _lease_s(self) -> float:
        return self._config.scheduler_config.tars.delivery_lease_s

    def _sync_topology(self) -> dict | None:
        instances = [
            instance
            for instance in self._instance_provider.get_available_instances(None).values()
            if PDRole(instance.role) in (PDRole.ROLE_P, PDRole.ROLE_D, PDRole.ROLE_U)
        ]
        target_map: dict[TarsTarget, tuple[Instance, Endpoint]] = {}
        fingerprint = []
        for instance in sorted(instances, key=lambda item: item.id):
            for endpoint in sorted(instance.get_all_endpoints(), key=lambda item: item.id):
                target = TarsTarget(str(instance.id), endpoint.id)
                if target in target_map:
                    return self._error("duplicate_target", f"duplicate target {target.key}")
                target_map[target] = (instance, endpoint)
                fingerprint.append(
                    (
                        target.key,
                        str(instance.role),
                        str(instance.engine_type or "").lower(),
                        endpoint.ip,
                        endpoint.business_port,
                    )
                )
        current_fingerprint = tuple(fingerprint)
        if current_fingerprint != self._topology_fingerprint:
            self._topology_epoch += 1
            specs = tuple(
                TarsReplicaSpec(
                    target=target,
                    pending_token_capacity=self._config.scheduler_config.tars.pending_token_capacity,
                    running_block_capacity=self._config.scheduler_config.tars.running_block_capacity,
                )
                for target in sorted(target_map)
            )
            self._runtime.sync_topology(self._topology_epoch, specs)
            self._topology_fingerprint = current_fingerprint
        self._target_map = target_map
        handoff_targets = tuple(
            HandoffTargetSpec(
                target=target,
                role=self._handoff_role(instance),
                engine_type=str(instance.engine_type or "").strip().lower(),
                circuit_open=self._circuit_open(instance.id),
            )
            for target, (instance, _endpoint) in sorted(target_map.items())
        )
        bootstrap_targets = tuple(
            BootstrapJointTargetSpec(
                target=target,
                role=(
                    BootstrapJointRole.DECODE if PDRole(instance.role) is PDRole.ROLE_D else BootstrapJointRole.PREFILL
                ),
                engine_type=str(instance.engine_type or "").strip().lower(),
                circuit_open=self._circuit_open(instance.id),
                decode_token_capacity=(
                    self._config.scheduler_config.tars.decode_token_capacity
                    if PDRole(instance.role) is PDRole.ROLE_D
                    else 0
                ),
            )
            for target, (instance, _endpoint) in sorted(target_map.items())
        )
        try:
            self._single.sync_targets(self._topology_epoch, handoff_targets)
            self._joint.sync_targets(self._topology_epoch, bootstrap_targets)
        except (TypeError, ValueError) as exc:
            return self._error("invalid_topology", str(exc))
        return None

    def _build_evidence(self, instances: list[Instance], data: dict) -> tuple[TarsCacheEvidence, ...] | None:
        token_ids = self._token_ids(data)
        if not instances:
            return None
        requests: list[TarsEvidenceRequest] = []
        expected: list[ExpectedDp] = []
        conductor_to_logical: dict[tuple[str, int], TarsTarget] = {}
        for instance in instances:
            conductor_id = conductor_instance_id(instance)
            for endpoint in instance.get_all_endpoints():
                logical = TarsTarget(str(instance.id), endpoint.id)
                conductor_key = (conductor_id, endpoint.id)
                conductor_to_logical[conductor_key] = logical
                requests.append(TarsEvidenceRequest(logical, len(token_ids), self._block_size))
                expected.append(
                    ExpectedDp(
                        instance_id=conductor_id,
                        dp_rank=endpoint.id,
                        model_name=instance.model_name,
                        tenant_id=TENANT_ID,
                        block_size=self._block_size,
                    )
                )
        try:
            workers = ConductorApiClient.get_registered_services()
            query = ConductorApiClient.query_conductor(instances, token_ids)
            source = evaluate_exact_dp_evidence(expected, {"workers": workers}, query)
        except (KeyError, TypeError, ValueError):
            return None
        if source.availability is EvidenceAvailability.UNAVAILABLE:
            return None
        translated: dict[tuple[str, int], ExactDpEvidence] = {}
        for conductor_key, item in source.tuples.items():
            logical = conductor_to_logical.get(conductor_key)
            if logical is None:
                return None
            translated[logical.key] = item
        batch = adapt_conductor_evidence(
            requests,
            ConductorEvidence(EvidenceAvailability.AVAILABLE, translated),
        )
        return batch.items if batch.availability is EvidenceAvailability.AVAILABLE else None

    def _eligible_instances(self, role: PDRole, engine_type: str) -> list[Instance]:
        normalized_engine = engine_type.strip().lower()
        return [
            instance
            for instance in self._instance_provider.get_available_instances(role).values()
            if not self._circuit_open(instance.id)
            and (not normalized_engine or str(instance.engine_type or "").strip().lower() == normalized_engine)
        ]

    def _attempt_spec(
        self,
        request_id: str,
        attempt_seq: int,
        data: dict,
        evidence: tuple[TarsCacheEvidence, ...],
    ) -> TarsAttemptSpec:
        context = TarsRequestContext(
            session_id=self._session_id(data, request_id),
            request_id=request_id,
            input_tokens=len(self._token_ids(data)),
        )
        from motor.coordinator.scheduler.policy.tars.state_runtime import TarsAttemptKey

        return TarsAttemptSpec(
            key=TarsAttemptKey(request_id, attempt_seq),
            request=context,
            evidence=evidence,
            max_output_tokens=self._nonnegative_int(data, "max_output_tokens"),
            block_size=self._block_size,
        )

    def _commit_actual(
        self,
        target: TarsTarget,
        role: PDRole,
        workload: Workload,
    ) -> _ActualCommit | None:
        mapped = self._target_map.get(target)
        if mapped is None:
            return None
        instance, endpoint = mapped
        receipt = self._workload_ledger.allocate(instance.id, endpoint.id, role, workload.active_tokens)
        if receipt is None:
            return None
        return _ActualCommit(target, role, workload, receipt)

    @staticmethod
    def _request_fingerprint(data: dict) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _replay_prepared(self, data: dict, kind: str) -> dict | None:
        key = self._key(self._required_text(data, "req_id"), self._positive_int(data, "attempt_seq"), kind)
        pending = self._actual.get(key)
        if pending is not None:
            if pending.releasing or pending.response is None:
                return self._error("workload_releasing", "Workload compensation is pending")
            if pending.request_fingerprint != self._request_fingerprint(data):
                return self._error("attempt_conflict", "Attempt input differs from its prepared delivery")
            return pending.response
        if key in self._settled:
            return self._error("attempt_settled", "Settled attempts cannot be admitted again")
        return None

    def _release_pending_actual(
        self,
        request_id: str,
        attempt_seq: int,
        kind: str,
    ) -> bool:
        key = self._key(request_id, attempt_seq, kind)
        pending = self._actual.get(key)
        if pending is None:
            return True
        pending.releasing = True
        released = [self._workload_ledger.release(commit.receipt) for commit in reversed(pending.commits)]
        if not all(released):
            return False
        self._actual.pop(key)
        pending.response = None
        self._settled[key] = pending
        if len(self._settled) > 4096:
            self._settled.popitem(last=False)
        return True

    def _select_decode(
        self,
        _admission: BootstrapJointAdmissionSpec,
        candidates: tuple[BootstrapDecodeCandidate, ...],
    ) -> TarsTarget:
        if not candidates:
            raise ValueError("no compatible Decode target")
        return min(candidates, key=lambda item: (item.committed_tokens, item.target)).target

    def _decision(self, event: str, data: dict, **details) -> None:
        if not self._config.scheduler_config.tars.enable_decision_log:
            return
        record = {
            "event": event,
            "req_id": data.get("req_id"),
            "attempt_seq": data.get("attempt_seq"),
            "scheduler_epoch": data.get("scheduler_epoch", self._scheduler_epoch),
            **details,
        }
        logger.info(
            "TARS decision %s",
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        )

    @staticmethod
    def _handoff_role(instance: Instance) -> HandoffRole:
        role = PDRole(instance.role)
        if role is PDRole.ROLE_P:
            return HandoffRole.PREFILL
        if role is PDRole.ROLE_D:
            return HandoffRole.DECODE
        return HandoffRole.UNIFIED

    @staticmethod
    def _resource_dict(instance: Instance, endpoint: Endpoint, workload: Workload) -> dict:
        return {
            "instance": instance.model_dump(mode="json"),
            "endpoint": endpoint.model_dump(mode="json"),
            "workload": workload.model_dump(mode="json"),
        }

    @staticmethod
    def _identity_dict(identity) -> dict:
        return {
            "request_id": identity.request_id,
            "attempt_seq": identity.attempt_seq,
            "scheduler_epoch": identity.scheduler_epoch,
        }

    def _delivery_input(self, data: dict) -> tuple[str, tuple[str, int, int], str]:
        kind = str(data.get("kind") or "")
        if kind not in ("single", "joint"):
            raise ValueError("kind must be single or joint")
        identity = (
            self._required_text(data, "req_id"),
            self._positive_int(data, "attempt_seq"),
            self._positive_or_zero_int(data, "scheduler_epoch"),
        )
        return kind, identity, self._required_text(data, "delivery_token")

    @staticmethod
    def _key(request_id: str, attempt_seq: int, kind: str) -> tuple[str, int, str]:
        return (request_id, attempt_seq, kind)

    @staticmethod
    def _error(code: str, message: str) -> dict:
        return {"error_code": code, "error": message}

    @staticmethod
    def _protocol_error(failure) -> dict:
        if failure is None:
            return {"error_code": "protocol_error", "error": "TARS protocol failed"}
        return {"error_code": failure.code.value, "error": failure.message}

    @staticmethod
    def _required_text(data: dict, name: str) -> str:
        value = data.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be non-empty")
        return value

    @staticmethod
    def _positive_int(data: dict, name: str) -> int:
        value = data.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _positive_or_zero_int(data: dict, name: str) -> int:
        value = data.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @staticmethod
    def _nonnegative_int(data: dict, name: str) -> int:
        value = data.get(name, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @staticmethod
    def _token_ids(data: dict) -> list[int]:
        token_ids = data.get("token_ids")
        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in token_ids):
            raise ValueError("token_ids must contain integers")
        return token_ids

    @staticmethod
    def _session_id(data: dict, request_id: str) -> str:
        value = data.get("session_id")
        return value.strip() if isinstance(value, str) and value.strip() else request_id


def evidence_by_target(
    evidence: tuple[TarsCacheEvidence, ...],
) -> dict[TarsTarget, TarsCacheEvidence]:
    """Return a typed lookup without leaking mutable protocol state."""
    return {item.target: item for item in evidence}
