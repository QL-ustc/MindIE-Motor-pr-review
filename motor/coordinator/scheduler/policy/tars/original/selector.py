# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Stable dual-hash and source-TARS cache/TTFT decision branch."""

import hashlib
from collections.abc import Callable, Iterable
from types import MappingProxyType

from motor.coordinator.scheduler.policy.tars.original.models import (
    TarsCandidate,
    TarsCandidateScore,
    TarsOriginalConfig,
    TarsRequestContext,
    TarsSelection,
    TarsTarget,
)
from motor.coordinator.scheduler.policy.tars.original.scorer import score_tars_candidate


def _hash_score(session_key: str, target: TarsTarget, *, person: bytes) -> int:
    payload = f"{session_key}\x1f{target.instance_id}\x1f{target.dp_rank}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16, person=person).digest(), "big")


def stable_dual_hash_targets(
    session_id: str,
    targets: Iterable[TarsTarget],
) -> tuple[TarsTarget, TarsTarget]:
    """Choose two stable logical targets without process hash or RNG state."""
    target_list = sorted(set(targets))
    if not target_list:
        raise ValueError("at least one TARS target is required")
    session_key = session_id.split("/", 1)[0]
    primary = max(
        target_list,
        key=lambda target: (_hash_score(session_key, target, person=b"tars-dh-primary"), target),
    )
    if len(target_list) == 1:
        return primary, primary
    second = max(
        (target for target in target_list if target != primary),
        key=lambda target: (_hash_score(session_key, target, person=b"tars-dh-second"), target),
    )
    return primary, second


def _order_pair(
    pair: tuple[TarsCandidate, TarsCandidate],
    metric: Callable[[TarsCandidate], float | int],
    *,
    primary_is_max: bool,
) -> tuple[TarsCandidate, TarsCandidate]:
    first, second = pair
    first_metric = metric(first)
    second_metric = metric(second)
    if first_metric != second_metric:
        first_wins = first_metric > second_metric if primary_is_max else first_metric < second_metric
        return (first, second) if first_wins else (second, first)
    if first.running_request_blocks != second.running_request_blocks:
        return (first, second) if first.running_request_blocks < second.running_request_blocks else (second, first)
    return pair


def select_tars_candidate(
    request: TarsRequestContext,
    candidates: Iterable[TarsCandidate],
    config: TarsOriginalConfig,
) -> TarsSelection | None:
    """Apply the original cache-first, overload-to-min-TTFT decision branch."""
    candidate_list = tuple(candidates)
    if not candidate_list:
        return None
    candidate_by_target = {candidate.target: candidate for candidate in candidate_list}
    if len(candidate_by_target) != len(candidate_list):
        raise ValueError("TARS candidate targets must be unique")

    first_target, second_target = stable_dual_hash_targets(request.session_id, candidate_by_target)
    pair = (candidate_by_target[first_target], candidate_by_target[second_target])
    unique_pair = {candidate.target: candidate for candidate in pair}
    scores: dict[TarsTarget, TarsCandidateScore] = {
        target: score_tars_candidate(request, candidate, config) for target, candidate in unique_pair.items()
    }

    high_credit, low_credit = _order_pair(
        pair,
        lambda candidate: candidate.best_effort_cache_credit_tokens,
        primary_is_max=True,
    )
    threshold = config.effective_first_balance_threshold_tokens * config.effective_tpct
    if scores[high_credit.target].predicted_ttft > threshold:
        primary, second = _order_pair(
            pair,
            lambda candidate: scores[candidate.target].predicted_ttft,
            primary_is_max=False,
        )
        decision_type = f"{config.balance_type}:min_global_ttft"
    else:
        primary, second = high_credit, low_credit
        decision_type = config.balance_type

    return TarsSelection(
        primary_target=primary.target,
        second_target=second.target,
        score_by_target=MappingProxyType(scores),
        decision_type=decision_type,
    )
