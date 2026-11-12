# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Reject legacy PyMotor names newly introduced by the TARS migration diff."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess

LEGACY_PATTERN = re.compile(r"(?i)(mindie[-_/ ]?pymotor|\bpymotor\b)")
SELF_TEST_PATHS = frozenset(
    {
        "scripts/check_tars_migration_paths.py",
        "tests/scripts/test_check_tars_migration_paths.py",
    }
)


def added_legacy_lines(diff_text: str) -> list[str]:
    """Return added legacy references, excluding diff metadata and existing context."""
    violations: list[str] = []
    current_path: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_path = line.removeprefix("+++ b/")
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        if current_path in SELF_TEST_PATHS:
            continue
        if LEGACY_PATTERN.search(line[1:]):
            violations.append(line[1:])
    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Frozen migration base ref or SHA")
    args = parser.parse_args()
    git_executable = shutil.which("git")
    if git_executable is None:
        parser.error("git executable is required")
    result = subprocess.run(
        [git_executable, "diff", "--no-ext-diff", "--unified=0", args.base, "--"],
        check=True,
        capture_output=True,
        text=True,
    )
    violations = added_legacy_lines(result.stdout)
    if not violations:
        return 0
    for line in violations:
        print(f"legacy TARS migration reference: {line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
