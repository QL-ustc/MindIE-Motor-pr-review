# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Strict Conductor evidence contract used by the dormant TARS adapter."""

import json
from pathlib import Path

import msgspec
import pytest

from motor.coordinator.api_client.conductor_evidence import (
    EvidenceAvailability,
    ExpectedDp,
    UnavailableEvidenceAction,
    evaluate_exact_dp_evidence,
)

EXPECTED = (
    ExpectedDp("vllm-prefill-1", 0, "qwen", "default", 128),
    ExpectedDp("vllm-prefill-1", 1, "qwen", "default", 128),
)
QUERY_FIXTURES = Path(__file__).resolve().parents[3] / "motor/kv_conductor/tests/fixtures"


def _workers(*, dp0_provenance=True, dp1_provenance=False):
    return {
        "workers": [
            {
                "instance_id": "vllm-prefill-1",
                "model_name": "qwen",
                "tenant_id": "default",
                "block_size": 128,
                "endpoints": {
                    "0": {
                        "medium_endpoints": {"npu": "tcp://10.0.0.1:50090"},
                        "engine_type": "vllm",
                        "dp_rank": 0,
                        "replay_endpoint": None,
                        "exact_dp_npu_provenance": dp0_provenance,
                    },
                    "1": {
                        "medium_endpoints": {"npu": "tcp://10.0.0.1:50091"},
                        "engine_type": "vllm",
                        "dp_rank": 1,
                        "replay_endpoint": None,
                        "exact_dp_npu_provenance": dp1_provenance,
                    },
                },
            }
        ]
    }


def _dp(*, matched=128, npu=1, provenance=True):
    return {
        "matched_tokens": matched,
        "npu_blocks": npu,
        "cpu_blocks": 0,
        "disk_blocks": 0,
        "exact_dp_npu_provenance": provenance,
    }


@pytest.mark.parametrize("encoding", ["json", "msgpack"])
def test_query_wire_fixture_preserves_exact_dp_credit(encoding):
    """Share the full wire contract with the Rust response serializer test."""
    payload = (QUERY_FIXTURES / f"query_response.{encoding}").read_bytes()
    response = json.loads(payload) if encoding == "json" else msgspec.msgpack.decode(payload)
    expected = tuple(ExpectedDp("prefill-0", rank, "qwen", "default", 128) for rank in (0, 1))
    workers = _workers(dp1_provenance=True)
    workers["workers"][0]["instance_id"] = "prefill-0"

    result = evaluate_exact_dp_evidence(expected, workers, response)

    assert result.availability is EvidenceAvailability.AVAILABLE
    assert result.tuples[("prefill-0", 0)].creditable_npu_blocks == 3
    assert result.tuples[("prefill-0", 0)].matched_tokens == 384
    assert result.tuples[("prefill-0", 0)].cpu_blocks == 0
    assert result.tuples[("prefill-0", 1)].creditable_npu_blocks == 0
    assert result.tuples[("prefill-0", 1)].cpu_blocks == 3


@pytest.mark.parametrize("field", ["npu_end", "cpu_end", "disk_end"])
@pytest.mark.parametrize("value", [-1, True, None, "1"])
def test_invalid_coverage_metadata_fails_closed(field, value):
    response = {
        "default": {
            "vllm-prefill-1": {
                "longest_matched": 128,
                "DP": {"0": {**_dp(), field: value}},
            }
        }
    }

    result = evaluate_exact_dp_evidence(EXPECTED, _workers(), response)

    assert result.availability is EvidenceAvailability.UNAVAILABLE
    assert result.reason == "invalid_query_schema"
    assert result.tuples == {}


@pytest.mark.parametrize("field", ["matched_tokens", "npu_blocks", "cpu_blocks", "disk_blocks"])
def test_missing_required_count_fails_closed(field):
    match = {**_dp(), "npu_end": 1, "cpu_end": 0, "disk_end": 0}
    del match[field]
    response = {"default": {"vllm-prefill-1": {"longest_matched": 128, "DP": {"0": match}}}}

    result = evaluate_exact_dp_evidence(EXPECTED, _workers(), response)

    assert result.availability is EvidenceAvailability.UNAVAILABLE
    assert result.reason == "invalid_query_schema"
    assert result.tuples == {}


def test_sparse_registered_tuple_is_available_zero_hit():
    result = evaluate_exact_dp_evidence(
        EXPECTED,
        _workers(),
        {"default": {"vllm-prefill-1": {"longest_matched": 128, "DP": {"0": _dp()}}}},
    )

    assert result.availability is EvidenceAvailability.AVAILABLE
    assert result.tuples[("vllm-prefill-1", 0)].npu_creditable is True
    assert result.tuples[("vllm-prefill-1", 1)].matched_tokens == 0
    assert result.tuples[("vllm-prefill-1", 1)].npu_creditable is False


def test_missing_or_false_provenance_is_valid_but_not_creditable():
    response = {
        "default": {
            "vllm-prefill-1": {
                "longest_matched": 128,
                "DP": {
                    "0": {
                        "matched_tokens": 128,
                        "npu_blocks": 1,
                        "cpu_blocks": 0,
                        "disk_blocks": 0,
                    },
                    "1": _dp(provenance=True),
                },
            }
        }
    }

    result = evaluate_exact_dp_evidence(EXPECTED, _workers(dp1_provenance=False), response)

    assert result.availability is EvidenceAvailability.AVAILABLE
    assert result.tuples[("vllm-prefill-1", 0)].npu_creditable is False
    assert result.tuples[("vllm-prefill-1", 0)].creditable_npu_blocks == 0
    assert result.tuples[("vllm-prefill-1", 1)].npu_creditable is False


def test_missing_registration_is_unavailable_without_partial_ranking():
    workers = _workers()
    del workers["workers"][0]["endpoints"]["1"]

    result = evaluate_exact_dp_evidence(
        EXPECTED,
        workers,
        {"default": {"vllm-prefill-1": {"longest_matched": 128, "DP": {"0": _dp()}}}},
    )

    assert result.availability is EvidenceAvailability.UNAVAILABLE
    assert result.tuples == {}
    assert result.reason == "missing_registration"


def test_mixed_namespace_is_unavailable():
    workers = _workers()
    workers["workers"][0]["model_name"] = "other-model"

    result = evaluate_exact_dp_evidence(EXPECTED, workers, {"default": {}})

    assert result.availability is EvidenceAvailability.UNAVAILABLE
    assert result.tuples == {}
    assert result.reason == "namespace_mismatch"


def test_legacy_or_mixed_query_schema_fails_closed():
    mixed = {
        "default": {
            "vllm-prefill-1": {
                "longest_matched": 128,
                "DP": {"0": {**_dp(), "NPU": 1}},
            }
        }
    }

    result = evaluate_exact_dp_evidence(EXPECTED, _workers(), mixed)

    assert result.availability is EvidenceAvailability.UNAVAILABLE
    assert result.tuples == {}
    assert result.reason == "invalid_query_schema"


def test_query_failure_is_unavailable():
    result = evaluate_exact_dp_evidence(EXPECTED, _workers(), None)

    assert result.availability is EvidenceAvailability.UNAVAILABLE
    assert result.reason == "query_failure"


def test_unavailable_fallback_boundary_is_explicit():
    result = evaluate_exact_dp_evidence(EXPECTED, _workers(), None)

    assert (
        result.unavailable_action(admission_started=False, tars_mutation_count=0)
        is UnavailableEvidenceAction.LOAD_BALANCE
    )
    assert (
        result.unavailable_action(admission_started=True, tars_mutation_count=0)
        is UnavailableEvidenceAction.STATUS_OR_REVOKE
    )
    assert (
        result.unavailable_action(admission_started=False, tars_mutation_count=None)
        is UnavailableEvidenceAction.STATUS_OR_REVOKE
    )
    assert (
        result.unavailable_action(admission_started=False, tars_mutation_count=1)
        is UnavailableEvidenceAction.STATUS_OR_REVOKE
    )
    assert (
        result.unavailable_action(admission_started=False, tars_mutation_count=False)
        is UnavailableEvidenceAction.STATUS_OR_REVOKE
    )


def test_available_evidence_never_requests_fallback():
    result = evaluate_exact_dp_evidence(EXPECTED, _workers(), {"default": {}})

    assert result.availability is EvidenceAvailability.AVAILABLE
    assert (
        result.unavailable_action(admission_started=False, tars_mutation_count=0)
        is UnavailableEvidenceAction.NOT_APPLICABLE
    )
