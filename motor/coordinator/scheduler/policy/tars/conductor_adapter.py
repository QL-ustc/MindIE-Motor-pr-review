# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Convert K3 Conductor evidence into provenance-gated TARS token accounting."""

from collections.abc import Iterable

from motor.coordinator.api_client.conductor_evidence import ConductorEvidence, EvidenceAvailability
from motor.coordinator.scheduler.policy.tars.models import (
    CacheCreditCapability,
    TarsCacheEvidence,
    TarsEvidenceBatch,
    TarsEvidenceRequest,
)


def adapt_conductor_evidence(
    requests: Iterable[TarsEvidenceRequest],
    conductor: ConductorEvidence,
) -> TarsEvidenceBatch:
    """Adapt one indivisible exact-DP evidence snapshot.

    CPU and Disk counts remain diagnostics.  Only NPU blocks carrying explicit
    server-owned exact-DP provenance receive best-effort scheduling credit.
    """
    request_list = tuple(requests)
    request_keys = [request.target.key for request in request_list]
    if not request_list or len(request_keys) != len(set(request_keys)):
        raise ValueError("TARS evidence requests must be non-empty and unique")

    if conductor.availability is EvidenceAvailability.UNAVAILABLE:
        return TarsEvidenceBatch(EvidenceAvailability.UNAVAILABLE, reason=conductor.reason)

    if set(request_keys) != set(conductor.tuples):
        return TarsEvidenceBatch(EvidenceAvailability.UNAVAILABLE, reason="evidence_tuple_mismatch")

    adapted = []
    for request in sorted(request_list, key=lambda item: item.target):
        source = conductor.tuples[request.target.key]
        capability = CacheCreditCapability.DP_NPU_ONLY if source.npu_creditable else CacheCreditCapability.COMPUTE_ONLY
        observed_tokens = (
            min(request.input_tokens, source.npu_blocks * request.block_size) if source.npu_creditable else 0
        )
        adapted.append(
            TarsCacheEvidence(
                target=request.target,
                input_tokens=request.input_tokens,
                capability=capability,
                observed_exact_dp_npu_tokens=observed_tokens,
                best_effort_cache_credit_tokens=observed_tokens,
                estimated_compute_tokens=request.input_tokens - observed_tokens,
                diagnostic_matched_tokens=source.matched_tokens,
                diagnostic_cpu_blocks=source.cpu_blocks,
                diagnostic_disk_blocks=source.disk_blocks,
            )
        )
    return TarsEvidenceBatch(EvidenceAvailability.AVAILABLE, items=tuple(adapted))
