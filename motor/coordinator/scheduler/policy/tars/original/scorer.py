# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Pure source-TARS TTFT cost formula using current neutral token names."""

from motor.coordinator.scheduler.policy.tars.original.models import (
    TarsCandidate,
    TarsCandidateScore,
    TarsOriginalConfig,
    TarsRequestContext,
)


def estimate_ttft(
    compute_tokens: float,
    best_effort_cache_credit_tokens: float,
    *,
    config: TarsOriginalConfig,
) -> float:
    """Estimate TTFT cost; cache credit is observed exact-DP NPU reuse only."""
    if compute_tokens < 0 or best_effort_cache_credit_tokens < 0:
        raise ValueError("TARS token costs must be non-negative")
    return round(
        compute_tokens * config.effective_tpct + best_effort_cache_credit_tokens * config.tprt,
        4,
    )


def score_tars_candidate(
    request: TarsRequestContext,
    candidate: TarsCandidate,
    config: TarsOriginalConfig,
) -> TarsCandidateScore:
    if candidate.estimated_compute_tokens + candidate.best_effort_cache_credit_tokens != request.input_tokens:
        raise ValueError("candidate compute and cache credit must account for every input token")

    compute_tokens = (
        candidate.global_waiting_compute_tokens + candidate.pending_compute_tokens + candidate.estimated_compute_tokens
    )
    cache_credit_tokens = (
        candidate.global_waiting_best_effort_cache_credit_tokens
        + candidate.pending_best_effort_cache_credit_tokens
        + candidate.best_effort_cache_credit_tokens
    )
    return TarsCandidateScore(
        target=candidate.target,
        predicted_ttft=estimate_ttft(compute_tokens, cache_credit_tokens, config=config),
        estimated_compute_tokens=candidate.estimated_compute_tokens,
        best_effort_cache_credit_tokens=candidate.best_effort_cache_credit_tokens,
    )
