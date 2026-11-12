# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Tests for the TARS legacy-name migration gate."""

from scripts.check_tars_migration_paths import added_legacy_lines


def test_gate_ignores_upstream_context_and_diff_metadata():
    diff = (
        "--- a/docker/MindIE-PyMotor/Dockerfile\n"
        "+++ b/docker/MindIE-PyMotor/Dockerfile\n"
        " MindIE-PyMotor existing context\n"
        "+MindIE Motor current name"
    )

    assert added_legacy_lines(diff) == []


def test_gate_rejects_new_legacy_product_names():
    diff = "+clone MindIE-PyMotor\n+WORKDIR /opt/pyMotor\n"

    assert added_legacy_lines(diff) == ["clone MindIE-PyMotor", "WORKDIR /opt/pyMotor"]


def test_gate_allows_only_its_scanner_and_fixture_paths():
    diff = (
        "diff --git a/scripts/check_tars_migration_paths.py b/scripts/check_tars_migration_paths.py\n"
        "+++ b/scripts/check_tars_migration_paths.py\n"
        '+LEGACY_PATTERN = re.compile(r"pymotor")\n'
        "diff --git a/docs/new.md b/docs/new.md\n"
        "+++ b/docs/new.md\n"
        "+new MindIE-PyMotor instruction\n"
    )

    assert added_legacy_lines(diff) == ["new MindIE-PyMotor instruction"]
