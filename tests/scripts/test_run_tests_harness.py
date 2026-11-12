# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the License at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = PROJECT_ROOT / "tests" / "run_tests.sh"
GITLEAKS_RUNNER_PATH = PROJECT_ROOT / "pre-commit" / "run_gitleaks.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _prepare_fake_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    tests_dir = repo / "tests"
    scripts_dir = repo / "scripts"
    bin_dir = tmp_path / "bin"
    tests_dir.mkdir(parents=True)
    scripts_dir.mkdir()
    bin_dir.mkdir()
    shutil.copy2(RUNNER_PATH, tests_dir / "run_tests.sh")

    python_calls = tmp_path / "python_calls.log"
    pytest_args = tmp_path / "pytest_args.log"
    proto_args = tmp_path / "proto_args.log"
    pip_called = tmp_path / "pip_called"

    _write_executable(
        bin_dir / "python3",
        """#!/bin/bash
printf '%s\n' "$@" >> "$PYTHON_CALLS_LOG"
if [ "${1:-}" = "-c" ]; then
    if [ "${FAKE_RUNTIME_FAILURE:-0}" = "1" ] && [[ "${2:-}" == *"readline"* ]]; then
        exit 139
    fi
    if [ -n "${FAKE_MISSING_IMPORT:-}" ] && [[ "${2:-}" == *"$FAKE_MISSING_IMPORT"* ]]; then
        exit 1
    fi
    exit 0
fi
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "pytest" ]; then
    printf '%s\n' "$@" > "$PYTEST_ARGS_LOG"
    exit "${FAKE_PYTEST_EXIT:-0}"
fi
exit 0
""",
    )
    _write_executable(
        bin_dir / "pip",
        """#!/bin/bash
touch "$PIP_CALLED_MARKER"
exit 99
""",
    )
    _write_executable(
        scripts_dir / "generate_proto.sh",
        """#!/bin/bash
printf '%s\n' "$@" > "$PROTO_ARGS_LOG"
if [ "${1:-}" != "--output-dir" ] || [ -z "${2:-}" ]; then
    exit 2
fi
mkdir -p "$2"
touch "$2/generated.marker"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "PYTHON_BIN": str(bin_dir / "python3"),
            "PYTHON_CALLS_LOG": str(python_calls),
            "PYTEST_ARGS_LOG": str(pytest_args),
            "PROTO_ARGS_LOG": str(proto_args),
            "PIP_CALLED_MARKER": str(pip_called),
        }
    )
    return repo, env


def _run_runner(repo: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(repo / "tests" / "run_tests.sh"), *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_has_no_cleanup_or_dependency_side_effects(tmp_path: Path) -> None:
    repo, env = _prepare_fake_repo(tmp_path)
    coverage_file = repo / ".coverage"
    htmlcov_sentinel = repo / "htmlcov" / "sentinel"
    coverage_file.touch()
    htmlcov_sentinel.parent.mkdir()
    htmlcov_sentinel.touch()

    result = _run_runner(repo, env, "--help")

    assert result.returncode == 0
    assert coverage_file.exists()
    assert htmlcov_sentinel.exists()
    assert not Path(env["PYTHON_CALLS_LOG"]).exists()
    assert not Path(env["PROTO_ARGS_LOG"]).exists()


def test_invalid_runner_argument_has_no_cleanup_or_dependency_side_effects(tmp_path: Path) -> None:
    repo, env = _prepare_fake_repo(tmp_path)
    coverage_file = repo / ".coverage"
    htmlcov_sentinel = repo / "htmlcov" / "sentinel"
    coverage_file.touch()
    htmlcov_sentinel.parent.mkdir()
    htmlcov_sentinel.touch()

    result = _run_runner(repo, env, "-n")

    assert result.returncode == 2
    assert "-n requires a process count" in result.stderr
    assert coverage_file.exists()
    assert htmlcov_sentinel.exists()
    assert not Path(env["PYTHON_CALLS_LOG"]).exists()
    assert not Path(env["PROTO_ARGS_LOG"]).exists()


def test_missing_dependency_fails_without_installing_or_generating(tmp_path: Path) -> None:
    repo, env = _prepare_fake_repo(tmp_path)
    env["FAKE_MISSING_IMPORT"] = "xdist"

    result = _run_runner(repo, env, "tests/example.py")

    assert result.returncode != 0
    assert "Missing required Python dependencies" in result.stderr
    assert "pytest-xdist" in result.stderr
    assert not Path(env["PIP_CALLED_MARKER"]).exists()
    assert not Path(env["PROTO_ARGS_LOG"]).exists()
    assert not Path(env["PYTEST_ARGS_LOG"]).exists()


def test_serial_run_does_not_require_xdist(tmp_path: Path) -> None:
    repo, env = _prepare_fake_repo(tmp_path)
    env["FAKE_MISSING_IMPORT"] = "xdist"

    result = _run_runner(repo, env, "--serial", "tests/example.py")

    assert result.returncode == 0
    assert Path(env["PROTO_ARGS_LOG"]).exists()
    assert Path(env["PYTEST_ARGS_LOG"]).exists()


def test_broken_python_runtime_fails_before_dependency_or_proto_work(tmp_path: Path) -> None:
    repo, env = _prepare_fake_repo(tmp_path)
    env["FAKE_RUNTIME_FAILURE"] = "1"

    result = _run_runner(repo, env, "--serial", "tests/example.py")

    assert result.returncode != 0
    assert "Python runtime sanity check failed" in result.stderr
    assert not Path(env["PIP_CALLED_MARKER"]).exists()
    assert not Path(env["PROTO_ARGS_LOG"]).exists()
    assert not Path(env["PYTEST_ARGS_LOG"]).exists()


def test_arguments_cleanup_and_pytest_exit_code_are_preserved(tmp_path: Path) -> None:
    repo, env = _prepare_fake_repo(tmp_path)
    env["FAKE_PYTEST_EXIT"] = "7"
    coverage_file = repo / ".coverage"
    htmlcov_sentinel = repo / "htmlcov" / "sentinel"
    coverage_file.touch()
    htmlcov_sentinel.parent.mkdir()
    htmlcov_sentinel.touch()

    result = _run_runner(repo, env, "--serial", "-k", "alpha or beta", "tests/example.py")

    assert result.returncode == 7
    assert coverage_file.exists()
    assert htmlcov_sentinel.exists()
    assert not Path(env["PIP_CALLED_MARKER"]).exists()
    pytest_args = Path(env["PYTEST_ARGS_LOG"]).read_text(encoding="utf-8").splitlines()
    assert pytest_args[:3] == ["-m", "pytest", "-o"]
    assert pytest_args[3].startswith("cache_dir=")
    assert pytest_args[4:] == ["--color=yes", "-k", "alpha or beta", "tests/example.py"]
    proto_args = Path(env["PROTO_ARGS_LOG"]).read_text(encoding="utf-8").splitlines()
    assert proto_args[0] == "--output-dir"
    generated_dir = Path(proto_args[1])
    assert not generated_dir.is_relative_to(repo)
    assert not generated_dir.exists()


def test_gitleaks_runner_fails_without_downloading_when_binary_is_missing(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["GITLEAKS_BIN"] = str(tmp_path / "missing-gitleaks")

    result = subprocess.run(
        ["bash", str(GITLEAKS_RUNNER_PATH), "protect", "--redact"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "gitleaks executable is not executable" in result.stderr
    assert "download" not in result.stdout.lower()


def test_gitleaks_runner_preserves_arguments_and_exit_code(tmp_path: Path) -> None:
    fake_gitleaks = tmp_path / "gitleaks"
    args_log = tmp_path / "args.log"
    _write_executable(
        fake_gitleaks,
        """#!/bin/bash
printf '%s\n' "$@" > "$GITLEAKS_ARGS_LOG"
exit 9
""",
    )
    env = os.environ.copy()
    env["GITLEAKS_BIN"] = str(fake_gitleaks)
    env["GITLEAKS_ARGS_LOG"] = str(args_log)

    result = subprocess.run(
        ["bash", str(GITLEAKS_RUNNER_PATH), "protect", "--redact", "path with spaces"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 9
    assert args_log.read_text(encoding="utf-8").splitlines() == ["protect", "--redact", "path with spaces"]
