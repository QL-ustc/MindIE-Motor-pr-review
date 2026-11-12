# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Strict, fail-closed interpretation of KV-Conductor scheduling evidence.

This module is intentionally independent from the active KVA policy.  It is
the protocol boundary consumed by the later TARS adapter: registered tuples
omitted from a valid sparse query are zero hits, while transport, namespace,
registration, and schema failures make the whole evidence set unavailable.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

TupleKey = tuple[str, int]


class EvidenceAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class UnavailableEvidenceAction(str, Enum):
    """Permitted action when Conductor evidence cannot be trusted."""

    NOT_APPLICABLE = "not_applicable"
    LOAD_BALANCE = "load_balance"
    STATUS_OR_REVOKE = "status_or_revoke"


@dataclass(frozen=True)
class ExpectedDp:
    instance_id: str
    dp_rank: int
    model_name: str
    tenant_id: str
    block_size: int

    @property
    def key(self) -> TupleKey:
        return (self.instance_id, self.dp_rank)


@dataclass(frozen=True)
class RegisteredDp:
    instance_id: str
    dp_rank: int
    model_name: str
    tenant_id: str
    block_size: int
    exact_dp_npu_provenance: bool


@dataclass(frozen=True)
class ExactDpEvidence:
    matched_tokens: int = 0
    npu_blocks: int = 0
    cpu_blocks: int = 0
    disk_blocks: int = 0
    npu_creditable: bool = False

    @property
    def creditable_npu_blocks(self) -> int:
        """NPU blocks eligible for exact-DP scheduling credit."""
        return self.npu_blocks if self.npu_creditable else 0


@dataclass(frozen=True)
class ConductorEvidence:
    availability: EvidenceAvailability
    tuples: dict[TupleKey, ExactDpEvidence] = field(default_factory=dict)
    reason: str = ""

    def unavailable_action(
        self,
        *,
        admission_started: bool,
        tars_mutation_count: int | None,
    ) -> UnavailableEvidenceAction:
        """Return the only safe next action at the scheduling boundary.

        Load-balancer fallback is allowed only before admission and when the
        caller can prove TARS has performed zero mutations.  An admitted or
        uncertain attempt must be reconciled through STATUS/revocation by the
        later stateful TARS layer.
        """
        if self.availability is EvidenceAvailability.AVAILABLE:
            return UnavailableEvidenceAction.NOT_APPLICABLE
        mutation_count_is_proven_zero = (
            isinstance(tars_mutation_count, int)
            and not isinstance(tars_mutation_count, bool)
            and tars_mutation_count == 0
        )
        if not admission_started and mutation_count_is_proven_zero:
            return UnavailableEvidenceAction.LOAD_BALANCE
        return UnavailableEvidenceAction.STATUS_OR_REVOKE


class ConductorSchemaError(ValueError):
    """The response is not the current KV-Conductor schema."""


_WORKER_KEYS = {"instance_id", "model_name", "tenant_id", "block_size", "endpoints"}
_ENDPOINT_KEYS = {
    "medium_endpoints",
    "engine_type",
    "dp_rank",
    "replay_endpoint",
    "exact_dp_npu_provenance",
}
_ENDPOINT_REQUIRED_KEYS = _ENDPOINT_KEYS - {"exact_dp_npu_provenance"}
_INSTANCE_MATCH_KEYS = {"longest_matched", "DP"}
_DP_MATCH_REQUIRED_KEYS = {
    "matched_tokens",
    "npu_blocks",
    "cpu_blocks",
    "disk_blocks",
}
# Absolute coverage ends are optional metadata; credit uses exclusive counts.
_DP_COVERAGE_KEYS = {"npu_end", "cpu_end", "disk_end"}
_DP_MATCH_KEYS = _DP_MATCH_REQUIRED_KEYS | _DP_COVERAGE_KEYS | {"exact_dp_npu_provenance"}


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def parse_registered_workers(
    response: Any,
) -> tuple[list[dict[str, Any]], dict[TupleKey, RegisteredDp]]:
    """Parse only ``GET /workers`` current lowercase schema.

    Missing provenance is accepted for rolling compatibility and interpreted
    as false.  Legacy Mooncake ``/services`` fields and mixed schemas are not
    normalized because doing so would manufacture trust across protocols.
    """
    if not isinstance(response, dict):
        raise ConductorSchemaError("workers response must be an object")
    response_keys = set(response)
    if "workers" not in response_keys or not response_keys <= {"workers", "indexer"}:
        raise ConductorSchemaError("workers response does not match the current lowercase schema")
    if "indexer" in response and not isinstance(response["indexer"], list):
        raise ConductorSchemaError("indexer summary must be a list")
    workers = response["workers"]
    if not isinstance(workers, list):
        raise ConductorSchemaError("workers must be a list")

    registered: dict[TupleKey, RegisteredDp] = {}
    for worker in workers:
        if not isinstance(worker, dict) or set(worker) != _WORKER_KEYS:
            raise ConductorSchemaError("worker entry does not match the current lowercase schema")
        instance_id = worker["instance_id"]
        model_name = worker["model_name"]
        tenant_id = worker["tenant_id"]
        block_size = worker["block_size"]
        endpoints = worker["endpoints"]
        if not all(isinstance(value, str) and value for value in (instance_id, model_name, tenant_id)):
            raise ConductorSchemaError("worker namespace strings must be non-empty")
        if not _is_nonnegative_int(block_size) or block_size == 0:
            raise ConductorSchemaError("worker block_size must be positive")
        if not isinstance(endpoints, dict):
            raise ConductorSchemaError("worker endpoints must be an object")

        for rank_text, endpoint in endpoints.items():
            if not isinstance(rank_text, str) or not rank_text.isdecimal():
                raise ConductorSchemaError("endpoint dp_rank key must be a decimal string")
            if not isinstance(endpoint, dict):
                raise ConductorSchemaError("endpoint must be an object")
            endpoint_keys = set(endpoint)
            if not _ENDPOINT_REQUIRED_KEYS <= endpoint_keys or not endpoint_keys <= _ENDPOINT_KEYS:
                raise ConductorSchemaError("endpoint does not match the current lowercase schema")
            dp_rank = endpoint["dp_rank"]
            if not _is_nonnegative_int(dp_rank) or dp_rank != int(rank_text):
                raise ConductorSchemaError("endpoint dp_rank does not match its map key")
            if not isinstance(endpoint["medium_endpoints"], dict) or not all(
                isinstance(medium, str) and isinstance(url, str) and medium and url
                for medium, url in endpoint["medium_endpoints"].items()
            ):
                raise ConductorSchemaError("medium_endpoints must map non-empty strings")
            if not isinstance(endpoint["engine_type"], str):
                raise ConductorSchemaError("engine_type must be a string")
            if endpoint["replay_endpoint"] is not None and not isinstance(endpoint["replay_endpoint"], str):
                raise ConductorSchemaError("replay_endpoint must be a string or null")
            provenance = endpoint.get("exact_dp_npu_provenance", False)
            if not isinstance(provenance, bool):
                raise ConductorSchemaError("exact_dp_npu_provenance must be boolean")

            key = (instance_id, dp_rank)
            if key in registered:
                raise ConductorSchemaError("duplicate worker tuple")
            registered[key] = RegisteredDp(
                instance_id=instance_id,
                dp_rank=dp_rank,
                model_name=model_name,
                tenant_id=tenant_id,
                block_size=block_size,
                exact_dp_npu_provenance=provenance,
            )
    return workers, registered


def _unavailable(reason: str) -> ConductorEvidence:
    return ConductorEvidence(EvidenceAvailability.UNAVAILABLE, reason=reason)


def evaluate_exact_dp_evidence(
    expected_dps: Iterable[ExpectedDp],
    workers_response: Any,
    query_response: Any,
) -> ConductorEvidence:
    """Validate registration and query evidence as one indivisible snapshot."""
    expected_list = tuple(expected_dps)
    expected = {item.key: item for item in expected_list}
    if not expected or len(expected) != len(expected_list):
        return _unavailable("invalid_expected_namespace")
    namespaces = {(item.model_name, item.tenant_id, item.block_size) for item in expected_list}
    if len(namespaces) != 1:
        return _unavailable("invalid_expected_namespace")
    _model_name, tenant_id, _block_size = next(iter(namespaces))

    try:
        _workers, registered = parse_registered_workers(workers_response)
    except ConductorSchemaError:
        return _unavailable("invalid_workers_schema")

    for key, item in expected.items():
        registration = registered.get(key)
        if registration is None:
            return _unavailable("missing_registration")
        if (
            registration.model_name != item.model_name
            or registration.tenant_id != item.tenant_id
            or registration.block_size != item.block_size
        ):
            return _unavailable("namespace_mismatch")

    if query_response is None:
        return _unavailable("query_failure")
    if not isinstance(query_response, dict) or set(query_response) != {tenant_id}:
        return _unavailable("invalid_query_schema")
    tenant_data = query_response[tenant_id]
    if not isinstance(tenant_data, dict):
        return _unavailable("invalid_query_schema")

    parsed: dict[TupleKey, ExactDpEvidence] = {}
    for instance_id, instance_data in tenant_data.items():
        if not isinstance(instance_id, str) or not isinstance(instance_data, dict):
            return _unavailable("invalid_query_schema")
        if set(instance_data) != _INSTANCE_MATCH_KEYS:
            return _unavailable("invalid_query_schema")
        if not _is_nonnegative_int(instance_data["longest_matched"]):
            return _unavailable("invalid_query_schema")
        dp_map = instance_data["DP"]
        if not isinstance(dp_map, dict):
            return _unavailable("invalid_query_schema")

        for rank_text, match in dp_map.items():
            if not isinstance(rank_text, str) or not rank_text.isdecimal() or not isinstance(match, dict):
                return _unavailable("invalid_query_schema")
            key = (instance_id, int(rank_text))
            if key not in expected:
                return _unavailable("namespace_mismatch")
            match_keys = set(match)
            if not _DP_MATCH_REQUIRED_KEYS <= match_keys or not match_keys <= _DP_MATCH_KEYS:
                return _unavailable("invalid_query_schema")
            numeric_keys = _DP_MATCH_REQUIRED_KEYS | (match_keys & _DP_COVERAGE_KEYS)
            if not all(_is_nonnegative_int(match[name]) for name in numeric_keys):
                return _unavailable("invalid_query_schema")
            response_provenance = match.get("exact_dp_npu_provenance", False)
            if not isinstance(response_provenance, bool):
                return _unavailable("invalid_query_schema")
            registration_provenance = registered[key].exact_dp_npu_provenance
            parsed[key] = ExactDpEvidence(
                matched_tokens=match["matched_tokens"],
                npu_blocks=match["npu_blocks"],
                cpu_blocks=match["cpu_blocks"],
                disk_blocks=match["disk_blocks"],
                npu_creditable=response_provenance and registration_provenance,
            )

    # A tuple known to be registered but omitted from a valid sparse response
    # is exactly a zero hit.  Capability remains server-owned via /workers.
    for key in expected:
        parsed.setdefault(
            key,
            ExactDpEvidence(npu_creditable=registered[key].exact_dp_npu_provenance),
        )
    return ConductorEvidence(EvidenceAvailability.AVAILABLE, tuples=parsed)
