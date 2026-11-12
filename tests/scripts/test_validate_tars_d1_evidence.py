# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Tests for the machine-readable TARS D1 evidence gate."""

import hashlib
import json

from scripts.validate_tars_d1_evidence import (
    parse_decision_log,
    validate_candidate_manifest,
    validate_decisions,
    validate_request_results,
    validate_workers,
)


def _record(event: str, **fields) -> dict:
    return {
        "event": event,
        "req_id": "request-1",
        "attempt_seq": 1,
        "scheduler_epoch": 11,
        **fields,
    }


def test_single_decision_lifecycle_and_results_pass(tmp_path):
    records = [
        _record("prepared", kind="single", target=["7", 1]),
        _record("confirmed", kind="single"),
        _record("lifecycle", kind="single", lifecycle_event="prefill_complete"),
        _record("lifecycle", kind="single", lifecycle_event="request_complete"),
    ]
    request_results = tmp_path / "results.jsonl"
    request_results.write_text(
        json.dumps({"req_id": "request-1", "status_code": 200}) + "\n",
        encoding="utf-8",
    )

    request_ids, errors = validate_decisions(records, 1, "handoff")

    assert errors == []
    assert validate_request_results(request_results, 1, request_ids) == []


def test_joint_decision_rejects_reuse_missing_confirm_and_bad_target():
    records = [
        _record(
            "prepared",
            kind="joint",
            prefill_target=["1", 0],
            decode_target="bad",
        ),
        _record("revoked", kind="joint"),
        {**_record("prepared", kind="joint", prefill_target=["1", 0], decode_target=["2", 0]), "scheduler_epoch": 12},
    ]

    _request_ids, errors = validate_decisions(records, 2, "bootstrap")

    assert any("invalid joint targets" in error for error in errors)
    assert any("has 0 confirmed records" in error for error in errors)
    assert any("reused across scheduler epochs" in error for error in errors)


def test_decision_rejects_terminal_before_confirmation():
    records = [
        _record("prepared", kind="single", target=["7", 1]),
        _record("lifecycle", kind="single", lifecycle_event="request_complete"),
        _record("confirmed", kind="single"),
    ]

    _request_ids, errors = validate_decisions(records, 1, "handoff")

    assert any("prepare < confirm < terminal" in error for error in errors)


def test_parser_extracts_json_and_rejects_legacy_text(tmp_path):
    log = tmp_path / "coordinator.log"
    log.write_text(
        "prefix TARS decision "
        + json.dumps(_record("confirmed", kind="single"))
        + "\nTARS decision event=prepared req_id=request-2\n",
        encoding="utf-8",
    )

    records, errors = parse_decision_log(log)

    assert records == [_record("confirmed", kind="single")]
    assert len(errors) == 1
    assert "invalid JSON" in errors[0]


def test_workers_require_exact_unique_tuples():
    snapshot = {
        "workers": [
            {
                "instance_id": "prefill-1",
                "endpoints": {
                    "0": {"dp_rank": 0, "exact_dp_npu_provenance": True},
                    "1": {"dp_rank": 1, "exact_dp_npu_provenance": False},
                },
            }
        ]
    }
    expected = [
        {
            "instance_id": "prefill-1",
            "dp_rank": 0,
            "require_exact_dp_npu_provenance": True,
        },
        {
            "instance_id": "prefill-1",
            "dp_rank": 1,
            "require_exact_dp_npu_provenance": True,
        },
    ]

    errors = validate_workers(snapshot, expected)

    assert errors == ["expected worker ('prefill-1', 1) lacks exact DP/NPU provenance"]


def test_workers_reject_endpoint_rank_mismatch_and_duplicate_expectation():
    snapshot = {
        "workers": [
            {
                "instance_id": "prefill-1",
                "endpoints": {"0": {"dp_rank": 1, "exact_dp_npu_provenance": True}},
            }
        ]
    }
    expected = [
        {"instance_id": "prefill-1", "dp_rank": 1},
        {"instance_id": "prefill-1", "dp_rank": 1},
    ]

    errors = validate_workers(snapshot, expected)

    assert any("endpoint key 0 does not match dp_rank 1" in error for error in errors)
    assert any("duplicate tuples" in error for error in errors)


def test_candidate_manifest_checks_head_hashes_versions_and_thresholds(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    checksum = hashlib.sha256(b"artifact").hexdigest()
    candidate_sha = "a" * 40
    manifest = {
        "final_candidate_sha": candidate_sha,
        "image_digest": "sha256:" + "b" * 64,
        "versions": {
            "motor": "3.1.0",
            "vllm": "test",
            "cann": "test",
            "driver": "test",
            "os": "test",
            "hardware": "test",
            "model": "test",
            "backend": "test",
        },
        "thresholds": {
            "conductor_query_p99_ms": 20,
            "non_tars_success_rate_drop_pp": 0.1,
            "cold_request_ttft_p95_regression_pct": 5,
        },
        "wheel_artifacts": [{"path": "artifact.bin", "sha256": checksum}],
        "config_artifacts": [{"path": "artifact.bin", "sha256": checksum}],
        "trace_artifacts": [{"path": "artifact.bin", "sha256": checksum}],
    }
    manifest_path = tmp_path / "candidate.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_candidate_manifest(manifest, manifest_path, candidate_sha) == []

    manifest["thresholds"]["cold_request_ttft_p95_regression_pct"] = 5.1
    errors = validate_candidate_manifest(manifest, manifest_path, "c" * 40)
    assert any("does not match repository HEAD" in error for error in errors)
    assert any("must be in [0, 5]" in error for error in errors)
