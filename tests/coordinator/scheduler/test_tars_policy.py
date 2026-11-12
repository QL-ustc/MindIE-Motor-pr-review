# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Golden and safety tests for the dormant pure TARS data path."""

import random
import threading
from collections import Counter
from unittest.mock import Mock

import pytest

from motor.common.resources.instance import InsStatus, Instance, PDRole
from motor.config.coordinator import SchedulerType
from motor.coordinator.api_client.conductor_api_client import ConductorApiClient
from motor.coordinator.api_client.conductor_evidence import (
    ConductorEvidence,
    EvidenceAvailability,
    ExactDpEvidence,
    UnavailableEvidenceAction,
)
from motor.coordinator.domain.agent_hint import AgentHintInfo
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.policy.factory import SchedulingPolicyFactory
from motor.coordinator.scheduler.policy.tars.adapter import (
    TarsRequestInputError,
    request_context_from_motor,
    target_from_motor,
)
from motor.coordinator.scheduler.policy.tars.conductor_adapter import adapt_conductor_evidence
from motor.coordinator.scheduler.policy.tars.models import (
    CacheCreditCapability,
    TarsEvidenceRequest,
    TarsTarget,
)
from motor.coordinator.scheduler.policy.tars.original.models import (
    TarsCandidate,
    TarsOriginalConfig,
    TarsRequestContext,
)
from motor.coordinator.scheduler.policy.tars.original.scorer import estimate_ttft
from motor.coordinator.scheduler.policy.tars.original.selector import (
    select_tars_candidate,
    stable_dual_hash_targets,
)
from motor.coordinator.scheduler.policy.tars.policy import TarsPolicy


def _request(
    *,
    request_id: str = "request-7",
    session_id: str | None = "agent-session-42",
    token_ids: list[int] | None = None,
    request_length: int = 9999,
) -> RequestInfo:
    return RequestInfo(
        req_id=request_id,
        req_data={"prompt": "ignored", "hash_session_id": "legacy-must-not-win"},
        req_len=request_length,
        token_ids=list(range(7)) if token_ids is None else token_ids,
        api="v1/completions",
        agent_hint_info=AgentHintInfo(session_id=session_id),
    )


def _target(instance_id: str, dp_rank: int) -> TarsTarget:
    return TarsTarget(instance_id=instance_id, dp_rank=dp_rank)


def _candidate(
    target: TarsTarget,
    *,
    compute_tokens: int,
    credit_tokens: int,
    pending_compute_tokens: float = 0,
    running_blocks: float = 0,
) -> TarsCandidate:
    return TarsCandidate(
        target=target,
        estimated_compute_tokens=compute_tokens,
        best_effort_cache_credit_tokens=credit_tokens,
        pending_compute_tokens=pending_compute_tokens,
        running_request_blocks=running_blocks,
    )


def test_request_adapter_prefers_agent_session_and_uses_token_count():
    request = _request(token_ids=[10, 20, 30], request_length=100_000)

    context = request_context_from_motor(request)

    assert context.session_id == "agent-session-42"
    assert context.request_id == "request-7"
    assert context.input_tokens == 3


def test_request_adapter_uses_explicit_request_id_fallback_and_rejects_missing_tokens():
    request = _request(session_id=None, token_ids=[1, 2])
    assert request_context_from_motor(request).session_id == request.req_id

    request.token_ids = None
    with pytest.raises(TarsRequestInputError, match="token_ids"):
        request_context_from_motor(request)


def test_target_identity_uses_instance_and_dp_only():
    instance = Instance(
        job_name="prefill-9",
        model_name="model-a",
        id=9,
        role=PDRole.ROLE_P,
        status=InsStatus.ACTIVE,
    )

    first = target_from_motor(instance, dp_rank=3)
    instance.job_name = "renamed"
    instance.model_name = "transport-metadata-changed"
    second = target_from_motor(instance, dp_rank=3)

    assert first == second == TarsTarget(instance_id="9", dp_rank=3)


@pytest.mark.parametrize(
    ("compute_tokens", "credit_tokens", "config", "expected"),
    [
        (10, 0, TarsOriginalConfig(tpct=0.5, tprt=0.2), 5.0),
        (4, 6, TarsOriginalConfig(tpct=0.5, tprt=0.2), 3.2),
        (0, 10, TarsOriginalConfig(tpct=0.5, tprt=0.0), 0.0),
    ],
)
def test_original_tars_scorer_golden_vectors(
    compute_tokens: int,
    credit_tokens: int,
    config: TarsOriginalConfig,
    expected: float,
):
    assert estimate_ttft(compute_tokens, credit_tokens, config=config) == expected


def test_original_tars_prefers_cache_credit_when_not_overloaded():
    target_a = _target("instance-a", 0)
    target_b = _target("instance-b", 0)
    request = TarsRequestContext(session_id="session/a", request_id="request-a", input_tokens=10)
    config = TarsOriginalConfig(tpct=1.0, tprt=0.0, dh_first_balance_ttft_threshold=5)
    candidates = [
        _candidate(target_a, compute_tokens=1, credit_tokens=9),
        _candidate(target_b, compute_tokens=5, credit_tokens=5),
    ]

    selection = select_tars_candidate(request, candidates, config)

    assert selection is not None
    assert selection.primary_target == target_a
    assert selection.second_target == target_b
    assert selection.decision_type == "tars"
    assert selection.primary_score.predicted_ttft == 1.0
    assert selection.primary_score.best_effort_cache_credit_tokens == 9
    with pytest.raises(TypeError):
        selection.score_by_target[target_a] = selection.primary_score


def test_original_tars_uses_min_ttft_when_high_credit_target_is_overloaded():
    target_a = _target("instance-a", 0)
    target_b = _target("instance-b", 0)
    request = TarsRequestContext(session_id="session/a", request_id="request-a", input_tokens=10)
    config = TarsOriginalConfig(tpct=1.0, tprt=0.0, dh_first_balance_ttft_threshold=5)
    candidates = [
        _candidate(target_a, compute_tokens=1, credit_tokens=9, pending_compute_tokens=10),
        _candidate(target_b, compute_tokens=5, credit_tokens=5),
    ]

    selection = select_tars_candidate(request, candidates, config)

    assert selection is not None
    assert selection.primary_target == target_b
    assert selection.second_target == target_a
    assert selection.decision_type == "tars:min_global_ttft"
    assert selection.score_by_target[target_a].predicted_ttft == 11.0
    assert selection.score_by_target[target_b].predicted_ttft == 5.0


def test_original_tars_single_candidate_keeps_source_decision_branch():
    target = _target("instance-a", 0)
    request = TarsRequestContext(session_id="session/a", request_id="request-a", input_tokens=10)
    config = TarsOriginalConfig(tpct=1.0, tprt=0.0, dh_first_balance_ttft_threshold=5)
    candidates = [
        _candidate(target, compute_tokens=1, credit_tokens=9, pending_compute_tokens=10),
    ]

    selection = select_tars_candidate(request, candidates, config)

    assert selection is not None
    assert selection.primary_target == target
    assert selection.second_target == target
    assert selection.decision_type == "tars:min_global_ttft"
    assert selection.primary_score.predicted_ttft == 11.0


def test_stable_dual_hash_is_order_independent_and_single_target_safe():
    targets = [_target("instance-b", 1), _target("instance-a", 0), _target("instance-a", 1)]

    first = stable_dual_hash_targets("session/root", targets)
    for _ in range(20):
        assert stable_dual_hash_targets("session/root", list(reversed(targets))) == first
    assert first == (_target("instance-a", 1), _target("instance-b", 1))
    assert first[0] != first[1]
    assert stable_dual_hash_targets("session/root", [targets[0]]) == (targets[0], targets[0])


def test_stable_dual_hash_distributes_10000_sessions_without_identity_collapse():
    targets = [_target(f"instance-{index // 4}", index % 4) for index in range(8)]
    counts = Counter(stable_dual_hash_targets(f"session-{index}", targets)[0] for index in range(10_000))

    assert set(counts) == set(targets)
    assert max(counts.values()) / 10_000 < 0.15
    assert min(counts.values()) / 10_000 > 0.10


def test_conductor_adapter_credits_only_proven_exact_dp_npu_blocks():
    exact = _target("instance-a", 0)
    ip_only = _target("instance-a", 1)
    requests = (
        TarsEvidenceRequest(target=exact, input_tokens=10, block_size=4),
        TarsEvidenceRequest(target=ip_only, input_tokens=10, block_size=4),
    )
    conductor = ConductorEvidence(
        EvidenceAvailability.AVAILABLE,
        tuples={
            exact.key: ExactDpEvidence(
                matched_tokens=10,
                npu_blocks=3,
                cpu_blocks=9,
                disk_blocks=8,
                npu_creditable=True,
            ),
            ip_only.key: ExactDpEvidence(
                matched_tokens=10,
                npu_blocks=3,
                cpu_blocks=9,
                disk_blocks=8,
                npu_creditable=False,
            ),
        },
    )

    batch = adapt_conductor_evidence(requests, conductor)
    by_target = batch.by_target

    assert batch.availability is EvidenceAvailability.AVAILABLE
    assert by_target[exact].capability is CacheCreditCapability.DP_NPU_ONLY
    assert by_target[exact].observed_exact_dp_npu_tokens == 10
    assert by_target[exact].best_effort_cache_credit_tokens == 10
    assert by_target[exact].estimated_compute_tokens == 0
    assert by_target[ip_only].capability is CacheCreditCapability.COMPUTE_ONLY
    assert by_target[ip_only].observed_exact_dp_npu_tokens == 0
    assert by_target[ip_only].best_effort_cache_credit_tokens == 0
    assert by_target[ip_only].estimated_compute_tokens == 10
    assert by_target[ip_only].diagnostic_cpu_blocks == 9
    assert by_target[ip_only].diagnostic_disk_blocks == 8


def test_conductor_adapter_gives_cpu_disk_only_evidence_zero_scheduling_credit():
    target = _target("instance-a", 0)
    request = TarsEvidenceRequest(target=target, input_tokens=10, block_size=4)
    conductor = ConductorEvidence(
        EvidenceAvailability.AVAILABLE,
        tuples={
            target.key: ExactDpEvidence(
                matched_tokens=10,
                npu_blocks=0,
                cpu_blocks=3,
                disk_blocks=4,
                npu_creditable=True,
            )
        },
    )

    result = adapt_conductor_evidence((request,), conductor).items[0]

    assert result.capability is CacheCreditCapability.DP_NPU_ONLY
    assert result.best_effort_cache_credit_tokens == 0
    assert result.estimated_compute_tokens == 10
    assert result.diagnostic_cpu_blocks == 3
    assert result.diagnostic_disk_blocks == 4


def test_conductor_adapter_distinguishes_sparse_zero_from_unavailable_without_partial_ranking():
    first = _target("instance-a", 0)
    second = _target("instance-b", 0)
    requests = (
        TarsEvidenceRequest(target=first, input_tokens=8, block_size=4),
        TarsEvidenceRequest(target=second, input_tokens=8, block_size=4),
    )
    sparse = ConductorEvidence(
        EvidenceAvailability.AVAILABLE,
        tuples={
            first.key: ExactDpEvidence(npu_creditable=True),
            second.key: ExactDpEvidence(npu_creditable=True),
        },
    )
    sparse_batch = adapt_conductor_evidence(requests, sparse)

    assert sparse_batch.availability is EvidenceAvailability.AVAILABLE
    assert all(item.best_effort_cache_credit_tokens == 0 for item in sparse_batch.items)

    incomplete = ConductorEvidence(
        EvidenceAvailability.AVAILABLE,
        tuples={first.key: ExactDpEvidence(npu_blocks=2, npu_creditable=True)},
    )
    unavailable_batch = adapt_conductor_evidence(requests, incomplete)

    assert unavailable_batch.availability is EvidenceAvailability.UNAVAILABLE
    assert unavailable_batch.items == ()
    assert unavailable_batch.reason == "evidence_tuple_mismatch"
    assert (
        unavailable_batch.unavailable_action(
            admission_started=False,
            tars_mutation_count=0,
        )
        is UnavailableEvidenceAction.LOAD_BALANCE
    )
    assert (
        unavailable_batch.unavailable_action(
            admission_started=True,
            tars_mutation_count=0,
        )
        is UnavailableEvidenceAction.STATUS_OR_REVOKE
    )

    transport_failure = ConductorEvidence(
        EvidenceAvailability.UNAVAILABLE,
        tuples={first.key: ExactDpEvidence(npu_blocks=2, npu_creditable=True)},
        reason="query_failure",
    )
    failed_batch = adapt_conductor_evidence(requests, transport_failure)
    assert failed_batch.availability is EvidenceAvailability.UNAVAILABLE
    assert failed_batch.items == ()
    assert failed_batch.reason == "query_failure"


def test_10000_generated_evidence_cases_have_zero_false_exact_dp_attribution():
    rng = random.Random(20260828)
    for index in range(10_000):
        target = _target(f"instance-{index % 17}", index % 4)
        input_tokens = rng.randrange(0, 32_768)
        block_size = rng.choice((16, 32, 64, 128))
        npu_blocks = rng.randrange(0, 1024)
        provenance = bool(rng.getrandbits(1))
        request = TarsEvidenceRequest(target=target, input_tokens=input_tokens, block_size=block_size)
        conductor = ConductorEvidence(
            EvidenceAvailability.AVAILABLE,
            tuples={
                target.key: ExactDpEvidence(
                    matched_tokens=rng.randrange(0, 32_768),
                    npu_blocks=npu_blocks,
                    cpu_blocks=rng.randrange(0, 1024),
                    disk_blocks=rng.randrange(0, 1024),
                    npu_creditable=provenance,
                )
            },
        )

        result = adapt_conductor_evidence((request,), conductor).items[0]
        expected_credit = min(input_tokens, npu_blocks * block_size) if provenance else 0

        assert result.observed_exact_dp_npu_tokens == expected_credit
        assert result.best_effort_cache_credit_tokens == expected_credit
        assert result.estimated_compute_tokens + result.best_effort_cache_credit_tokens == input_tokens
        if not provenance:
            assert result.capability is CacheCreditCapability.COMPUTE_ONLY


def test_dormant_tars_import_and_construction_have_no_runtime_side_effects(monkeypatch):
    provider = Mock()
    registry_before = dict(SchedulingPolicyFactory.create.__globals__["_REGISTRY"])
    query_mock = Mock(side_effect=AssertionError("network call"))
    thread_start_mock = Mock(side_effect=AssertionError("background task"))
    monkeypatch.setattr(ConductorApiClient, "query_conductor", query_mock)
    monkeypatch.setattr(threading.Thread, "start", thread_start_mock)

    policy = TarsPolicy(instance_provider=provider)
    load_balance = SchedulingPolicyFactory.create(SchedulerType.LOAD_BALANCE, provider)

    assert policy.select_pure is not None
    assert load_balance is not None
    assert SchedulingPolicyFactory.create.__globals__["_REGISTRY"] == registry_before
    query_mock.assert_not_called()
    thread_start_mock.assert_not_called()
