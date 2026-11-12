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

# Generate Python modules from the etcd .proto files. The default output stays
# beside the sources for packaging. Tests use --output-dir to avoid source-tree
# artifacts.

set -e

show_help() {
    echo "Usage: $0 [--output-dir DIR]"
    echo "  --output-dir DIR  Write generated modules to DIR instead of the source tree"
}

OUTPUT_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --output-dir)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo "Error: --output-dir requires a directory." >&2
                exit 2
            fi
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            show_help >&2
            exit 2
            ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="$ROOT_DIR/motor/common/etcd/proto"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c "import grpc_tools.protoc" 2>/dev/null; then
    echo "Error: grpcio-tools is not installed for $PYTHON_BIN." >&2
    echo "Install the project requirements explicitly before generating protobuf files." >&2
    exit 1
fi

PROTO_FILES=("$PROTO_DIR"/*.proto)
if [ ! -e "${PROTO_FILES[0]}" ]; then
    echo "No .proto files found in $PROTO_DIR."
    exit 0
fi

rewrite_generated_imports() {
    local generated_file="$1"
    local package_path="$2"
    "$PYTHON_BIN" - "$generated_file" "$package_path" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
package = sys.argv[2]
text = path.read_text(encoding="utf-8")
pattern = re.compile(r"^import ([A-Za-z0-9_]+_pb2)( as [A-Za-z0-9_]+)?$", re.MULTILINE)
text = pattern.sub(lambda match: f"from {package} import {match.group(1)}{match.group(2) or ''}", text)
path.write_text(text, encoding="utf-8")
PY
}

PACKAGE_PATH="motor.common.etcd.proto"
for proto_file in "${PROTO_FILES[@]}"; do
    proto_base="$(basename "$proto_file" .proto)"
    target_dir="${OUTPUT_DIR:-$PROTO_DIR}"
    mkdir -p "$target_dir"
    echo "Generating code from $proto_file into $target_dir..."

    "$PYTHON_BIN" -m grpc_tools.protoc \
        --proto_path="$PROTO_DIR" \
        --python_out="$target_dir" \
        --grpc_python_out="$target_dir" \
        "$(basename "$proto_file")"

    pb2_file="$target_dir/${proto_base}_pb2.py"
    pb2_grpc_file="$target_dir/${proto_base}_pb2_grpc.py"
    [ ! -f "$pb2_file" ] || rewrite_generated_imports "$pb2_file" "$PACKAGE_PATH"
    [ ! -f "$pb2_grpc_file" ] || rewrite_generated_imports "$pb2_grpc_file" "$PACKAGE_PATH"
done

echo "All protobuf files generated successfully."
