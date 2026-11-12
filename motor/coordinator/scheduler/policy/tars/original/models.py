# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Immutable inputs and outputs for the pure TARS selector."""

import math
from collections.abc import Mapping
from dataclasses import dataclass


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_nonnegative_number(value: object) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0
    )


@dataclass(frozen=True, order=True)
class TarsTarget:
    """Logical scheduler identity; transport endpoint data is intentionally absent."""

    instance_id: str
    dp_rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id:
            raise ValueError("instance_id must be a non-empty string")
        if not _is_nonnegative_int(self.dp_rank):
            raise ValueError("dp_rank must be a non-negative integer")

    @property
    def key(self) -> tuple[str, int]:
        return (self.instance_id, self.dp_rank)


@dataclass(frozen=True)
class TarsOriginalConfig:
    """Source TARS cost and first-balance parameters without runtime knobs."""

    balance_type: str = "tars"
    prefill_tpot: float = 0.000127
    tpct: float | None = None
    tprt: float = 0.0
    ttft_slo: float = 5.0
    dh_first_balance_ttft_threshold: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.balance_type, str) or not self.balance_type:
            raise ValueError("balance_type must be a non-empty string")
        if not _is_nonnegative_number(self.effective_tpct) or self.effective_tpct == 0:
            raise ValueError("tpct must be finite and positive")
        if not _is_nonnegative_number(self.tprt):
            raise ValueError("tprt must be finite and non-negative")
        if not _is_nonnegative_number(self.ttft_slo) or self.ttft_slo == 0:
            raise ValueError("ttft_slo must be finite and positive")
        if self.dh_first_balance_ttft_threshold is not None and not _is_nonnegative_number(
            self.dh_first_balance_ttft_threshold
        ):
            raise ValueError("dh_first_balance_ttft_threshold must be finite and non-negative")

    @property
    def effective_tpct(self) -> float:
        return self.prefill_tpot if self.tpct is None else self.tpct

    @property
    def effective_first_balance_threshold_tokens(self) -> float:
        if self.dh_first_balance_ttft_threshold is not None:
            return self.dh_first_balance_ttft_threshold
        return int((self.ttft_slo / 2) * (1 / self.effective_tpct))


@dataclass(frozen=True)
class TarsRequestContext:
    session_id: str
    request_id: str
    input_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not _is_nonnegative_int(self.input_tokens):
            raise ValueError("input_tokens must be a non-negative integer")


@dataclass(frozen=True)
class TarsCandidate:
    target: TarsTarget
    estimated_compute_tokens: int
    best_effort_cache_credit_tokens: int
    global_waiting_compute_tokens: float = 0.0
    global_waiting_best_effort_cache_credit_tokens: float = 0.0
    pending_compute_tokens: float = 0.0
    pending_best_effort_cache_credit_tokens: float = 0.0
    running_request_blocks: float = 0.0

    def __post_init__(self) -> None:
        if not _is_nonnegative_int(self.estimated_compute_tokens):
            raise ValueError("estimated_compute_tokens must be a non-negative integer")
        if not _is_nonnegative_int(self.best_effort_cache_credit_tokens):
            raise ValueError("best_effort_cache_credit_tokens must be a non-negative integer")
        costs = (
            self.global_waiting_compute_tokens,
            self.global_waiting_best_effort_cache_credit_tokens,
            self.pending_compute_tokens,
            self.pending_best_effort_cache_credit_tokens,
            self.running_request_blocks,
        )
        if not all(_is_nonnegative_number(value) for value in costs):
            raise ValueError("candidate queue and running costs must be finite and non-negative")


@dataclass(frozen=True)
class TarsCandidateScore:
    target: TarsTarget
    predicted_ttft: float
    estimated_compute_tokens: int
    best_effort_cache_credit_tokens: int


@dataclass(frozen=True)
class TarsSelection:
    primary_target: TarsTarget
    second_target: TarsTarget
    score_by_target: Mapping[TarsTarget, TarsCandidateScore]
    decision_type: str

    @property
    def primary_score(self) -> TarsCandidateScore:
        return self.score_by_target[self.primary_target]
