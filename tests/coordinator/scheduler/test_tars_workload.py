# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#         http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Exercise TARS receipts at the native API boundary, without loading a Rust library."""

from types import SimpleNamespace
from unittest.mock import call, create_autospec

import pytest

from motor.common.resources.instance import PDRole
from motor.coordinator.scheduler.runtime.tars_workload import (
    TarsWorkloadLedger,
    TarsWorkloadReceipt,
)
from motor.coordinator.scheduler.runtime.workload_shm.layout import (
    FLAG_BLOCKED,
    FLAG_VALID,
    ROLE_DECODE,
    ROLE_HYBRID,
    ROLE_PREFILL,
    SCHEMA_VERSION,
)
from motor.coordinator.scheduler.runtime.workload_shm.native import (
    STATUS_BAD_ARG,
    STATUS_BLOCKED,
    STATUS_CHANGED,
    STATUS_OK,
    STATUS_SLOT_INVALID,
    NativeWorkloadShmError,
    WorkloadShm,
)


def _native(role=ROLE_PREFILL):
    native = create_autospec(WorkloadShm, instance=True, spec_set=True)
    native.read_header.return_value = {
        "schema_version": SCHEMA_VERSION,
        "sequence": 2,
        "entry_count": 1,
    }
    native.load_entries.return_value = [
        {
            "slot": 0,
            "instance_id": 101,
            "endpoint_id": 201,
            "role": role,
            "flags": FLAG_VALID,
            "generation": 7,
            "active_tokens": 11.0,
            "request_count": 2,
            "total_requests": 5,
        }
    ]
    native.cas_add.return_value = (STATUS_OK, 18.0)
    native.cas_sub_floor0.return_value = (STATUS_OK, 11.0)
    return native


@pytest.mark.parametrize(
    "role,role_code",
    [
        (PDRole.ROLE_P, ROLE_PREFILL),
        (PDRole.ROLE_D, ROLE_DECODE),
        (PDRole.ROLE_U, ROLE_HYBRID),
    ],
)
def test_allocation_and_release_use_original_identity_and_generation(role, role_code):
    native = _native(role_code)
    ledger = TarsWorkloadLedger(SimpleNamespace(native=native))

    receipt = ledger.allocate(101, 201, role, 7.0)

    assert receipt == TarsWorkloadReceipt(101, 201, role, 7, 7.0)
    native.cas_add.assert_called_once_with(101, 201, 7, 11.0, 7.0)
    # A moved pair must be released by identity even while a new snapshot is unavailable.
    native.load_entries.side_effect = NativeWorkloadShmError("membership is being rebuilt")
    assert ledger.release(receipt) is True
    native.cas_sub_floor0.assert_called_once_with(101, 201, 7, 7.0)


def test_changed_counter_retries_with_fresh_expected_without_new_generation():
    native = _native()
    native.cas_add.side_effect = [(STATUS_CHANGED, 23.0), (STATUS_OK, 30.0)]
    ledger = TarsWorkloadLedger(SimpleNamespace(native=native))

    receipt = ledger.allocate(101, 201, PDRole.ROLE_P, 7.0)

    assert receipt is not None
    assert receipt.generation == 7
    assert native.cas_add.call_args_list == [
        call(101, 201, 7, 11.0, 7.0),
        call(101, 201, 7, 23.0, 7.0),
    ]


def test_generation_retired_during_cas_retry_cannot_allocate_replacement():
    native = _native()
    native.cas_add.side_effect = [(STATUS_CHANGED, 23.0), (STATUS_SLOT_INVALID, 99.0)]
    ledger = TarsWorkloadLedger(SimpleNamespace(native=native))

    assert ledger.allocate(101, 201, PDRole.ROLE_P, 7.0) is None
    assert native.cas_add.call_args_list == [
        call(101, 201, 7, 11.0, 7.0),
        call(101, 201, 7, 23.0, 7.0),
    ]
    native.cas_sub_floor0.assert_not_called()


@pytest.mark.parametrize("status", [STATUS_BLOCKED, STATUS_SLOT_INVALID, STATUS_BAD_ARG])
def test_native_allocation_refusal_does_not_create_receipt(status):
    native = _native()
    if status == STATUS_BLOCKED:
        native.load_entries.return_value[0]["flags"] |= FLAG_BLOCKED
    native.cas_add.return_value = (status, 11.0)
    ledger = TarsWorkloadLedger(SimpleNamespace(native=native))

    assert ledger.allocate(101, 201, PDRole.ROLE_P, 7.0) is None
    assert native.cas_add.call_count <= 1
    native.cas_sub_floor0.assert_not_called()


@pytest.mark.parametrize(
    "schema,sequences",
    [(SCHEMA_VERSION + 1, (2, 2)), (SCHEMA_VERSION, (1, 1)), (SCHEMA_VERSION, (2, 4))],
)
def test_incompatible_or_unstable_snapshot_cannot_allocate(schema, sequences):
    native = _native()
    native.read_header.side_effect = [
        {"schema_version": schema, "sequence": sequence, "entry_count": 1} for sequence in sequences
    ]
    ledger = TarsWorkloadLedger(SimpleNamespace(native=native))

    assert ledger.allocate(101, 201, PDRole.ROLE_P, 7.0) is None
    native.cas_add.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [("instance_id", 999), ("endpoint_id", 999), ("role", ROLE_DECODE), ("flags", 0)],
)
def test_snapshot_must_contain_the_exact_valid_role_target(field, value):
    native = _native()
    native.load_entries.return_value[0][field] = value
    ledger = TarsWorkloadLedger(SimpleNamespace(native=native))

    assert ledger.allocate(101, 201, PDRole.ROLE_P, 7.0) is None
    native.cas_add.assert_not_called()


@pytest.mark.parametrize("operation", ["read_header", "load_entries", "cas_add"])
def test_native_allocation_error_is_unavailable(operation):
    native = _native()
    getattr(native, operation).side_effect = NativeWorkloadShmError("native handle unavailable")
    ledger = TarsWorkloadLedger(SimpleNamespace(native=native))

    assert ledger.allocate(101, 201, PDRole.ROLE_P, 7.0) is None


def test_release_failure_remains_retryable_and_success_is_idempotent():
    native = _native()
    native.cas_sub_floor0.side_effect = [
        NativeWorkloadShmError("temporarily unavailable"),
        (STATUS_OK, 11.0),
    ]
    ledger = TarsWorkloadLedger(SimpleNamespace(native=native))
    receipt = TarsWorkloadReceipt(101, 201, PDRole.ROLE_P, 7, 7.0)

    assert ledger.release(receipt) is False
    assert receipt.released is False
    assert ledger.release(receipt) is True
    assert ledger.release(receipt) is True
    assert native.cas_sub_floor0.call_args_list == [
        call(101, 201, 7, 7.0),
        call(101, 201, 7, 7.0),
    ]


def test_native_release_error_status_does_not_settle_receipt():
    native = _native()
    native.cas_sub_floor0.side_effect = [(STATUS_BAD_ARG, 11.0), (STATUS_OK, 4.0)]
    ledger = TarsWorkloadLedger(SimpleNamespace(native=native))
    receipt = TarsWorkloadReceipt(101, 201, PDRole.ROLE_P, 7, 7.0)

    assert ledger.release(receipt) is False
    assert receipt.released is False
    assert ledger.release(receipt) is True
    assert native.cas_sub_floor0.call_count == 2


def test_retired_generation_is_settled_without_debiting_replacement():
    native = _native()
    ledger = TarsWorkloadLedger(SimpleNamespace(native=native))
    receipt = ledger.allocate(101, 201, PDRole.ROLE_P, 7.0)
    assert receipt is not None
    native.load_entries.return_value[0]["generation"] = 8
    native.cas_sub_floor0.return_value = (STATUS_SLOT_INVALID, 99.0)

    assert ledger.release(receipt) is True
    assert ledger.release(receipt) is True
    native.cas_sub_floor0.assert_called_once_with(101, 201, 7, 7.0)


def test_zero_token_receipt_still_accounts_for_one_native_request():
    native = _native()
    counters = {"active_tokens": 11.0, "request_count": 2, "total_requests": 5}

    def allocate(_instance_id, _endpoint_id, _generation, expected, delta, slot=None):
        assert slot is None
        assert expected == counters["active_tokens"]
        counters["active_tokens"] += delta
        counters["request_count"] += 1
        counters["total_requests"] += 1
        return STATUS_OK, counters["active_tokens"]

    def release(_instance_id, _endpoint_id, _generation, delta, slot=None):
        assert slot is None
        counters["active_tokens"] = max(0.0, counters["active_tokens"] - delta)
        counters["request_count"] = max(0, counters["request_count"] - 1)
        return STATUS_OK, counters["active_tokens"]

    native.cas_add.side_effect = allocate
    native.cas_sub_floor0.side_effect = release
    ledger = TarsWorkloadLedger(SimpleNamespace(native=native))

    receipt = ledger.allocate(101, 201, PDRole.ROLE_P, 0.0)

    assert receipt is not None
    assert counters == {"active_tokens": 11.0, "request_count": 3, "total_requests": 6}
    assert ledger.release(receipt) is True
    assert ledger.release(receipt) is True
    assert counters == {"active_tokens": 11.0, "request_count": 2, "total_requests": 6}
    native.cas_add.assert_called_once_with(101, 201, 7, 11.0, 0.0)
    native.cas_sub_floor0.assert_called_once_with(101, 201, 7, 0.0)


@pytest.mark.parametrize("owner", [None, SimpleNamespace(native=None)])
def test_absent_native_owner_keeps_release_pending(owner):
    ledger = TarsWorkloadLedger(owner)
    receipt = TarsWorkloadReceipt(101, 201, PDRole.ROLE_P, 7, 7.0)

    assert ledger.allocate(101, 201, PDRole.ROLE_P, 7.0) is None
    assert ledger.release(receipt) is False
    assert receipt.released is False
