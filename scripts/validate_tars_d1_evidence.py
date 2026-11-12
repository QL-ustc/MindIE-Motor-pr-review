# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Validate the machine-readable portion of a captured TARS D1 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DECISION_MARKER = "TARS decision "
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
REQUIRED_VERSIONS = (
    "motor",
    "vllm",
    "cann",
    "driver",
    "os",
    "hardware",
    "model",
    "backend",
)
SINGLE_MODES = frozenset({"handoff", "trigger", "pdhybrid"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_errors(artifacts: Any, manifest_dir: Path, field_name: str) -> list[str]:
    if not isinstance(artifacts, list) or not artifacts:
        return [f"{field_name} must contain at least one artifact"]
    errors: list[str] = []
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            errors.append(f"{field_name}[{index}] must be an object")
            continue
        path_value = artifact.get("path")
        expected = artifact.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            errors.append(f"{field_name}[{index}].path is required")
            continue
        path = (manifest_dir / path_value).resolve()
        if not path.is_file():
            errors.append(f"{field_name}[{index}] file not found: {path}")
            continue
        actual = _sha256(path)
        if expected != actual:
            errors.append(f"{field_name}[{index}] sha256 mismatch: expected={expected} actual={actual}")
    return errors


def validate_candidate_manifest(manifest: Any, manifest_path: Path, repo_head: str) -> list[str]:
    """Validate frozen candidate identity, artifacts, versions, and thresholds."""
    if not isinstance(manifest, dict):
        return ["candidate manifest must be an object"]
    errors: list[str] = []
    candidate_sha = manifest.get("final_candidate_sha")
    if not isinstance(candidate_sha, str) or SHA_PATTERN.fullmatch(candidate_sha) is None:
        errors.append("final_candidate_sha must be a lowercase 40-character Git SHA")
    elif candidate_sha != repo_head:
        errors.append(f"final_candidate_sha does not match repository HEAD: manifest={candidate_sha} head={repo_head}")
    image_digest = manifest.get("image_digest")
    if not isinstance(image_digest, str) or DIGEST_PATTERN.fullmatch(image_digest) is None:
        errors.append("image_digest must be an immutable sha256 digest")

    versions = manifest.get("versions")
    if not isinstance(versions, dict):
        errors.append("versions must be an object")
    else:
        for field in REQUIRED_VERSIONS:
            if not isinstance(versions.get(field), str) or not versions[field].strip():
                errors.append(f"versions.{field} is required")

    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, dict):
        errors.append("thresholds must be an object")
    else:
        query_p99 = thresholds.get("conductor_query_p99_ms")
        success_drop = thresholds.get("non_tars_success_rate_drop_pp")
        ttft_regression = thresholds.get("cold_request_ttft_p95_regression_pct")
        if not isinstance(query_p99, (int, float)) or query_p99 <= 0:
            errors.append("thresholds.conductor_query_p99_ms must be positive")
        if not isinstance(success_drop, (int, float)) or not 0 <= success_drop <= 0.1:
            errors.append("thresholds.non_tars_success_rate_drop_pp must be in [0, 0.1]")
        if not isinstance(ttft_regression, (int, float)) or not 0 <= ttft_regression <= 5:
            errors.append("thresholds.cold_request_ttft_p95_regression_pct must be in [0, 5]")

    manifest_dir = manifest_path.resolve().parent
    errors.extend(_artifact_errors(manifest.get("wheel_artifacts"), manifest_dir, "wheel_artifacts"))
    errors.extend(_artifact_errors(manifest.get("config_artifacts"), manifest_dir, "config_artifacts"))
    errors.extend(_artifact_errors(manifest.get("trace_artifacts"), manifest_dir, "trace_artifacts"))
    return errors


def parse_decision_log(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract compact JSON records from Coordinator logs."""
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if DECISION_MARKER not in line:
            continue
        payload = line.split(DECISION_MARKER, 1)[1].strip()
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as error:
            errors.append(f"decision log line {line_number} is invalid JSON: {error}")
            continue
        if not isinstance(record, dict):
            errors.append(f"decision log line {line_number} must contain an object")
            continue
        records.append(record)
    if not records:
        errors.append("decision log contains no TARS decision records")
    return records, errors


def _target_valid(target: Any) -> bool:
    return (
        isinstance(target, list)
        and len(target) == 2
        and isinstance(target[0], str)
        and bool(target[0])
        and isinstance(target[1], int)
        and not isinstance(target[1], bool)
        and target[1] >= 0
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_decisions(records: list[dict[str, Any]], expected_requests: int, mode: str) -> tuple[set[str], list[str]]:
    """Validate unique attempts, exact targets, confirmation, and terminal events."""
    errors: list[str] = []
    by_attempt: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    pair_epochs: dict[tuple[str, int], set[int]] = defaultdict(set)
    for record in records:
        req_id = record.get("req_id")
        attempt_seq = record.get("attempt_seq")
        scheduler_epoch = record.get("scheduler_epoch")
        if (
            not isinstance(req_id, str)
            or not req_id
            or not _positive_int(attempt_seq)
            or not _positive_int(scheduler_epoch)
        ):
            errors.append(f"decision record has invalid attempt identity: {record}")
            continue
        key = (req_id, attempt_seq, scheduler_epoch)
        by_attempt[key].append(record)
        if record.get("event") == "prepared":
            pair_epochs[(req_id, attempt_seq)].add(scheduler_epoch)

    prepared_keys: set[tuple[str, int, int]] = set()
    expected_kind = "joint" if mode == "bootstrap" else "single"
    for key, attempt_records in by_attempt.items():
        prepared = [record for record in attempt_records if record.get("event") == "prepared"]
        if not prepared:
            continue
        prepared_keys.add(key)
        if len(prepared) != 1:
            errors.append(f"attempt {key} has {len(prepared)} prepared records")
            continue
        prepared_record = prepared[0]
        if prepared_record.get("kind") != expected_kind:
            errors.append(f"attempt {key} kind does not match mode {mode}: {prepared_record.get('kind')}")
        if expected_kind == "single":
            if not _target_valid(prepared_record.get("target")):
                errors.append(f"attempt {key} has invalid single target")
        elif not _target_valid(prepared_record.get("prefill_target")) or not _target_valid(
            prepared_record.get("decode_target")
        ):
            errors.append(f"attempt {key} has invalid joint targets")

        confirmed = [record for record in attempt_records if record.get("event") == "confirmed"]
        if len(confirmed) != 1:
            errors.append(f"attempt {key} has {len(confirmed)} confirmed records")
        terminals = [
            record
            for record in attempt_records
            if record.get("event") == "revoked"
            or (record.get("event") == "lifecycle" and record.get("lifecycle_event") in ("request_complete", "abort"))
        ]
        if len(terminals) != 1:
            errors.append(f"attempt {key} has {len(terminals)} terminal records")
        if len(confirmed) == 1 and len(terminals) == 1:
            prepared_index = attempt_records.index(prepared_record)
            confirmed_index = attempt_records.index(confirmed[0])
            terminal_index = attempt_records.index(terminals[0])
            if not prepared_index < confirmed_index < terminal_index:
                errors.append(f"attempt {key} decision order is not prepare < confirm < terminal")

    if len(prepared_keys) != expected_requests:
        errors.append(f"prepared request count mismatch: expected={expected_requests} actual={len(prepared_keys)}")
    for pair, epochs in pair_epochs.items():
        if len(epochs) > 1:
            errors.append(f"attempt identity reused across scheduler epochs: {pair}")
    return {key[0] for key in prepared_keys}, errors


def validate_request_results(path: Path, expected_requests: int, prepared_req_ids: set[str]) -> list[str]:
    errors: list[str] = []
    request_ids: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"request result line {line_number} is invalid JSON: {error}")
            continue
        req_id = result.get("req_id") if isinstance(result, dict) else None
        status_code = result.get("status_code") if isinstance(result, dict) else None
        if not isinstance(req_id, str) or not req_id:
            errors.append(f"request result line {line_number} has invalid req_id")
            continue
        request_ids.append(req_id)
        if not isinstance(status_code, int) or not 200 <= status_code < 300:
            errors.append(f"request {req_id} did not return 2xx: status_code={status_code}")
    counts = Counter(request_ids)
    duplicates = sorted(req_id for req_id, count in counts.items() if count != 1)
    if duplicates:
        errors.append(f"request results contain duplicate ids: {duplicates}")
    if len(request_ids) != expected_requests:
        errors.append(f"request result count mismatch: expected={expected_requests} actual={len(request_ids)}")
    if set(request_ids) != prepared_req_ids:
        errors.append("request result ids do not exactly match prepared TARS request ids")
    return errors


def validate_workers(snapshot: Any, expected: Any) -> list[str]:
    """Require each expected Conductor `(instance_id, dp_rank)` exactly once."""
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("workers"), list):
        return ["workers snapshot must contain a workers list"]
    if not isinstance(expected, list) or not expected:
        return ["expected workers must be a non-empty list"]
    actual_counts: Counter[tuple[str, int]] = Counter()
    provenance: dict[tuple[str, int], bool] = {}
    errors: list[str] = []
    for worker in snapshot["workers"]:
        if not isinstance(worker, dict) or not isinstance(worker.get("endpoints"), dict):
            errors.append("workers snapshot contains an invalid worker object")
            continue
        instance_id = worker.get("instance_id")
        for endpoint_key, endpoint in worker["endpoints"].items():
            if not isinstance(instance_id, str) or not isinstance(endpoint, dict):
                errors.append("workers snapshot contains an invalid endpoint")
                continue
            dp_rank = endpoint.get("dp_rank")
            if not isinstance(dp_rank, int) or isinstance(dp_rank, bool) or dp_rank < 0:
                errors.append(f"worker {instance_id} has invalid dp_rank")
                continue
            if str(endpoint_key) != str(dp_rank):
                errors.append(f"worker {instance_id} endpoint key {endpoint_key} does not match dp_rank {dp_rank}")
            key = (instance_id, dp_rank)
            actual_counts[key] += 1
            provenance[key] = endpoint.get("exact_dp_npu_provenance") is True

    expected_keys: set[tuple[str, int]] = set()
    expected_counts: Counter[tuple[str, int]] = Counter()
    for item in expected:
        if not isinstance(item, dict):
            errors.append("expected workers contains a non-object entry")
            continue
        key = (item.get("instance_id"), item.get("dp_rank"))
        if (
            not isinstance(key[0], str)
            or not key[0]
            or not isinstance(key[1], int)
            or isinstance(key[1], bool)
            or key[1] < 0
        ):
            errors.append(f"invalid expected worker: {item}")
            continue
        expected_keys.add(key)
        expected_counts[key] += 1
        if actual_counts[key] != 1:
            errors.append(f"expected worker {key} occurs {actual_counts[key]} times")
        if item.get("require_exact_dp_npu_provenance") is True and not provenance.get(key, False):
            errors.append(f"expected worker {key} lacks exact DP/NPU provenance")
    unexpected = sorted(set(actual_counts) - expected_keys)
    if unexpected:
        errors.append(f"workers snapshot contains unexpected tuples: {unexpected}")
    duplicates = sorted(key for key, count in expected_counts.items() if count != 1)
    if duplicates:
        errors.append(f"expected workers contains duplicate tuples: {duplicates}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--decision-log", type=Path, required=True)
    parser.add_argument("--request-results", type=Path, required=True)
    parser.add_argument("--workers-snapshot", type=Path, required=True)
    parser.add_argument("--expected-workers", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, default=1000)
    parser.add_argument("--mode", choices=sorted(SINGLE_MODES | {"bootstrap"}), required=True)
    parser.add_argument("--topology", choices=("p4d4", "kv_pool_dp2"), required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    if args.expected_requests < 1000:
        errors.append("D1 functional evidence requires at least 1000 requests")
    git_executable = shutil.which("git")
    if git_executable is None:
        parser.error("git executable is required")
    head = subprocess.run(
        [git_executable, "-C", str(args.repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [git_executable, "-C", str(args.repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        errors.append("candidate repository must be clean before artifact fingerprints are frozen")
    manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    errors.extend(validate_candidate_manifest(manifest, args.candidate_manifest, head))
    records, parse_errors = parse_decision_log(args.decision_log)
    errors.extend(parse_errors)
    prepared_req_ids, decision_errors = validate_decisions(records, args.expected_requests, args.mode)
    errors.extend(decision_errors)
    errors.extend(validate_request_results(args.request_results, args.expected_requests, prepared_req_ids))
    snapshot = json.loads(args.workers_snapshot.read_text(encoding="utf-8"))
    expected = json.loads(args.expected_workers.read_text(encoding="utf-8"))
    errors.extend(validate_workers(snapshot, expected))

    report = {
        "status": "PASS" if not errors else "FAIL",
        "topology": args.topology,
        "mode": args.mode,
        "expected_requests": args.expected_requests,
        "prepared_requests": len(prepared_req_ids),
        "errors": errors,
        "scope": (
            "bundle integrity, decision lifecycle, HTTP results, and Conductor worker identity; "
            "not hardware, fault, performance, rollback, dispatch-target, or internal-quiescence proof"
        ),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(output)
    if args.report is not None:
        args.report.write_text(output + "\n", encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
