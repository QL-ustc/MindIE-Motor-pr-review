# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Dormant Motor policy wrapper around the pure TARS selector."""

from collections.abc import Iterable

from motor.common.resources.endpoint import Endpoint
from motor.common.resources.instance import Instance, PDRole
from motor.coordinator.domain import InstanceProvider
from motor.coordinator.scheduler.policy.base import BaseSchedulingPolicy
from motor.coordinator.scheduler.policy.tars.original.models import (
    TarsCandidate,
    TarsOriginalConfig,
    TarsRequestContext,
    TarsSelection,
)
from motor.coordinator.scheduler.policy.tars.original.selector import select_tars_candidate


class TarsPolicy(BaseSchedulingPolicy):
    """Pure wrapper intentionally absent from SchedulerType and the policy factory."""

    def __init__(
        self,
        instance_provider: InstanceProvider,
        config: TarsOriginalConfig | None = None,
    ) -> None:
        super().__init__(instance_provider)
        self._config = config or TarsOriginalConfig()

    def select_pure(
        self,
        request: TarsRequestContext,
        candidates: Iterable[TarsCandidate],
    ) -> TarsSelection | None:
        return select_tars_candidate(request, candidates, self._config)

    def _select_instance(self, role: PDRole | None = None) -> Instance | None:
        del role

    def _select_endpoint(self, instance: Instance) -> Endpoint | None:
        del instance
