#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -n "${GITLEAKS_BIN:-}" ]; then
    SELECTED_GITLEAKS="$GITLEAKS_BIN"
elif [ -x "$ROOT_DIR/gitleaks" ]; then
    SELECTED_GITLEAKS="$ROOT_DIR/gitleaks"
elif command -v gitleaks >/dev/null 2>&1; then
    SELECTED_GITLEAKS="$(command -v gitleaks)"
else
    echo "Error: gitleaks executable not found." >&2
    echo "Provide the approved offline binary at $ROOT_DIR/gitleaks or set GITLEAKS_BIN." >&2
    echo "This hook never downloads or installs gitleaks." >&2
    exit 1
fi

if [ ! -x "$SELECTED_GITLEAKS" ]; then
    echo "Error: configured gitleaks executable is not executable: $SELECTED_GITLEAKS" >&2
    exit 1
fi

exec "$SELECTED_GITLEAKS" "$@"
