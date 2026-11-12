# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Side-effect-free adaptation from current Motor request/topology objects."""

from motor.common.resources.instance import Instance
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.policy.tars.models import TarsTarget
from motor.coordinator.scheduler.policy.tars.original.models import TarsRequestContext


class TarsRequestInputError(ValueError):
    """A Motor request lacks the token or identity data required by TARS."""


def request_context_from_motor(request: RequestInfo) -> TarsRequestContext:
    """Resolve agent session identity and exact token count without tokenizing."""
    request_id = request.req_id
    if not isinstance(request_id, str) or not request_id:
        raise TarsRequestInputError("req_id must be a non-empty string")

    session_id = getattr(request.agent_hint_info, "session_id", None)
    if not isinstance(session_id, str) or not session_id.strip():
        session_id = request_id

    token_ids = request.token_ids
    if token_ids is None:
        raise TarsRequestInputError("token_ids are required; req_len is not a token count")
    if not isinstance(token_ids, list):
        raise TarsRequestInputError("token_ids must be a list")

    return TarsRequestContext(
        session_id=session_id,
        request_id=request_id,
        input_tokens=len(token_ids),
    )


def target_from_motor(instance: Instance, dp_rank: int) -> TarsTarget:
    """Build the sole logical identity without endpoint/IP/port metadata."""
    return TarsTarget(instance_id=str(instance.id), dp_rank=dp_rank)
