# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Generation-bound SHM receipts owned by the TARS management process."""

from dataclasses import dataclass

from motor.common.resources.instance import PDRole
from motor.coordinator.scheduler.runtime.workload_shm import WorkloadSharedMemoryOwner
from motor.coordinator.scheduler.runtime.workload_shm.layout import (
    FLAG_VALID,
    ROLE_DECODE,
    ROLE_HYBRID,
    ROLE_PREFILL,
    SCHEMA_VERSION,
)
from motor.coordinator.scheduler.runtime.workload_shm.native import (
    STATUS_CHANGED,
    STATUS_OK,
    STATUS_SLOT_INVALID,
    NativeWorkloadShmError,
)


@dataclass
class TarsWorkloadReceipt:
    """Retain the original allocation generation even after topology changes."""

    instance_id: int
    endpoint_id: int
    role: PDRole
    generation: int
    tokens: float
    released: bool = False


class TarsWorkloadLedger:
    """Use the same native atomic counters as ordinary Worker allocations."""

    def __init__(self, owner: WorkloadSharedMemoryOwner | None) -> None:
        self._owner = owner

    def allocate(self, instance_id: int, endpoint_id: int, role: PDRole, tokens: float) -> TarsWorkloadReceipt | None:
        native = self._owner.native if self._owner is not None else None
        if native is None:
            return None
        role_code = {PDRole.ROLE_P: ROLE_PREFILL, PDRole.ROLE_D: ROLE_DECODE, PDRole.ROLE_U: ROLE_HYBRID}[role]
        try:
            # Mgmt serializes membership changes; Workers may concurrently update counters.
            header = native.read_header()
            if header["schema_version"] != SCHEMA_VERSION or header["sequence"] & 1:
                return None
            entries = native.load_entries(header["entry_count"])
            if native.read_header()["sequence"] != header["sequence"]:
                return None
            entry = next(
                (
                    item
                    for item in entries
                    if item["instance_id"] == instance_id
                    and item["endpoint_id"] == endpoint_id
                    and item["role"] == role_code
                    and item["flags"] & FLAG_VALID
                ),
                None,
            )
            if entry is None:
                return None
            expected = entry["active_tokens"]
            for _ in range(64):
                status, actual = native.cas_add(instance_id, endpoint_id, entry["generation"], expected, tokens)
                if status == STATUS_OK:
                    return TarsWorkloadReceipt(instance_id, endpoint_id, role, entry["generation"], tokens)
                if status != STATUS_CHANGED:
                    return None
                expected = actual
        except NativeWorkloadShmError:
            return None
        return None

    def release(self, receipt: TarsWorkloadReceipt) -> bool:
        if receipt.released:
            return True
        native = self._owner.native if self._owner is not None else None
        if native is None:
            return False
        try:
            # Scan by identity: a live endpoint can move slots while retaining its generation.
            status, _actual = native.cas_sub_floor0(
                receipt.instance_id, receipt.endpoint_id, receipt.generation, receipt.tokens
            )
        except NativeWorkloadShmError:
            return False
        if status in (STATUS_OK, STATUS_SLOT_INVALID):
            # An absent/retired generation has no remaining ledger to debit.
            receipt.released = True
        return receipt.released
