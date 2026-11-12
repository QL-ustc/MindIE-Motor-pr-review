# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Current Motor identities and cache-evidence models used by dormant TARS."""

from dataclasses import dataclass
from enum import Enum

from motor.coordinator.api_client.conductor_evidence import (
    ConductorEvidence,
    EvidenceAvailability,
    UnavailableEvidenceAction,
)
from motor.coordinator.scheduler.policy.tars.original.models import TarsTarget


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class CacheCreditCapability(str, Enum):
    """Cache evidence that is safe to use in pure TARS scoring."""

    COMPUTE_ONLY = "compute_only"
    DP_NPU_ONLY = "dp_npu_only"


@dataclass(frozen=True)
class TarsEvidenceRequest:
    target: TarsTarget
    input_tokens: int
    block_size: int

    def __post_init__(self) -> None:
        if not _is_nonnegative_int(self.input_tokens):
            raise ValueError("input_tokens must be a non-negative integer")
        if not _is_nonnegative_int(self.block_size) or self.block_size == 0:
            raise ValueError("block_size must be a positive integer")


@dataclass(frozen=True)
class TarsCacheEvidence:
    target: TarsTarget
    input_tokens: int
    capability: CacheCreditCapability
    observed_exact_dp_npu_tokens: int
    best_effort_cache_credit_tokens: int
    estimated_compute_tokens: int
    diagnostic_matched_tokens: int
    diagnostic_cpu_blocks: int
    diagnostic_disk_blocks: int

    def __post_init__(self) -> None:
        counts = (
            self.input_tokens,
            self.observed_exact_dp_npu_tokens,
            self.best_effort_cache_credit_tokens,
            self.estimated_compute_tokens,
            self.diagnostic_matched_tokens,
            self.diagnostic_cpu_blocks,
            self.diagnostic_disk_blocks,
        )
        if not all(_is_nonnegative_int(value) for value in counts):
            raise ValueError("TARS evidence counters must be non-negative integers")
        if self.observed_exact_dp_npu_tokens != self.best_effort_cache_credit_tokens:
            raise ValueError("observed and best-effort cache credit must agree")
        if self.estimated_compute_tokens + self.best_effort_cache_credit_tokens != self.input_tokens:
            raise ValueError("compute and cache-credit tokens must equal input_tokens")
        if self.capability is CacheCreditCapability.COMPUTE_ONLY and self.best_effort_cache_credit_tokens != 0:
            raise ValueError("compute-only evidence cannot carry cache credit")


@dataclass(frozen=True)
class TarsEvidenceBatch:
    availability: EvidenceAvailability
    items: tuple[TarsCacheEvidence, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        targets = [item.target for item in self.items]
        if len(targets) != len(set(targets)):
            raise ValueError("TARS evidence targets must be unique")
        if self.availability is EvidenceAvailability.UNAVAILABLE and self.items:
            raise ValueError("unavailable evidence cannot expose partial candidates")

    @property
    def by_target(self) -> dict[TarsTarget, TarsCacheEvidence]:
        return {item.target: item for item in self.items}

    def unavailable_action(
        self,
        *,
        admission_started: bool,
        tars_mutation_count: int | None,
    ) -> UnavailableEvidenceAction:
        source = ConductorEvidence(self.availability, reason=self.reason)
        return source.unavailable_action(
            admission_started=admission_started,
            tars_mutation_count=tars_mutation_count,
        )
