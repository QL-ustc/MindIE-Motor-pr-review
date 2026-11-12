# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Router-side delivery reconciliation tests for public TARS activation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from motor.common.resources.endpoint import Endpoint, Workload, WorkloadAction
from motor.common.resources.instance import Instance, PDRole
from motor.config.coordinator import TarsConfig
from motor.coordinator.domain.agent_hint import AgentHintInfo
from motor.coordinator.domain.scheduling import UpdateWorkloadParams
from motor.coordinator.models.request import RequestInfo
from motor.coordinator.scheduler.runtime.scheduler_client import (
    AsyncSchedulerClient,
    SchedulerClientConfig,
)
from motor.coordinator.scheduler.runtime.zmq_protocol import (
    SchedulerRequestType,
    SchedulerResponse,
    SchedulerResponseType,
)


def _client(scheduler_type: str = "tars") -> AsyncSchedulerClient:
    return AsyncSchedulerClient(
        SchedulerClientConfig(
            scheduler_type=scheduler_type,
            tars=TarsConfig(),
        )
    )


def _response(data: dict | None = None, *, success: bool = True) -> SchedulerResponse:
    return SchedulerResponse(
        response_type=(SchedulerResponseType.SUCCESS if success else SchedulerResponseType.ERROR),
        request_id="server-response",
        data=data if data is not None else {"success": success},
        error=None if success else "injected",
    )


def _delivery() -> dict:
    return {
        "kind": "single",
        "identity": {
            "request_id": "request-1",
            "attempt_seq": 1,
            "scheduler_epoch": 9,
        },
        "delivery_token": "delivery-1",
    }


async def test_confirm_response_loss_reconciles_confirmed_server_state():
    client = _client()
    entry = client._persist_tars_delivery(_delivery())  # pylint: disable=protected-access
    client._transport.send_request = AsyncMock(  # pylint: disable=protected-access
        side_effect=[
            None,
            _response({"success": True, "confirmed": True, "state": "ownership_transferred"}),
        ]
    )

    confirmed = await client._confirm_tars_delivery(entry)  # pylint: disable=protected-access

    assert confirmed is True
    assert entry["confirmed"] is True
    assert client.tars_has_attempt("request-1", 1)


async def test_revoke_response_loss_reconciles_terminal_receipt_before_forgetting_ledger():
    client = _client()
    entry = client._persist_tars_delivery(_delivery())  # pylint: disable=protected-access
    client._transport.send_request = AsyncMock(  # pylint: disable=protected-access
        side_effect=[
            None,
            _response(
                {
                    "success": True,
                    "confirmed": False,
                    "state": "allocation_revoked",
                }
            ),
        ]
    )

    revoked = await client._revoke_tars_delivery(entry)  # pylint: disable=protected-access

    assert revoked is True
    assert not client.tars_has_attempt("request-1", 1)


async def test_ambiguous_revoke_keeps_router_ledger_for_later_reconciliation():
    client = _client()
    entry = client._persist_tars_delivery(_delivery())  # pylint: disable=protected-access
    client._transport.send_request = AsyncMock(return_value=None)  # pylint: disable=protected-access

    revoked = await client._revoke_tars_delivery(entry)  # pylint: disable=protected-access

    assert revoked is False
    assert client.tars_has_attempt("request-1", 1)


def test_non_tars_client_allocates_no_tars_router_state():
    client = _client("load_balance")

    assert client._tars_attempt_sequences is None  # pylint: disable=protected-access
    assert client._tars_ledger is None  # pylint: disable=protected-access
    assert client._tars_workload_receipts is None  # pylint: disable=protected-access
    assert not client.tars_has_attempt("request-1", 1)


@pytest.mark.parametrize(
    ("prefill", "decode"),
    [("tars", "load_balance"), ("round_robin", "tars")],
)
def test_client_rejects_unpaired_tars_before_creating_transport(prefill, decode, monkeypatch):
    """Programmatic client construction enforces the same boundary as config loading."""

    def unexpected_transport(*_args, **_kwargs):
        pytest.fail("Invalid TARS roles must be rejected before creating transport")

    monkeypatch.setattr(
        "motor.coordinator.scheduler.runtime.scheduler_client._SchedulerTransport",
        unexpected_transport,
    )
    with pytest.raises(ValueError, match="prefill_scheduler_type and decode_scheduler_type"):
        AsyncSchedulerClient(SchedulerClientConfig(prefill_scheduler_type=prefill, decode_scheduler_type=decode))


@pytest.mark.parametrize(
    ("prefill", "decode", "enabled"),
    [("tars", "tars", True), ("load_balance", "round_robin", False)],
)
def test_client_role_fields_control_joint_state_over_legacy_default(prefill, decode, enabled):
    client = AsyncSchedulerClient(
        SchedulerClientConfig(
            scheduler_type="tars",
            prefill_scheduler_type=prefill,
            decode_scheduler_type=decode,
        )
    )

    assert (client._tars_ledger is not None) is enabled  # pylint: disable=protected-access
    assert (client._tars_attempt_sequences is not None) is enabled  # pylint: disable=protected-access


def test_tars_decode_capacity_uses_output_token_limit_not_request_body_length():
    req_info = RequestInfo(
        req_id="request-1",
        req_data={"max_completion_tokens": 13},
        req_len=99_999,
        token_ids=[1, 2, 3],
        api="v1/chat/completions",
    )

    data = AsyncSchedulerClient._tars_request_data(  # pylint: disable=protected-access
        req_info,
        attempt_seq=1,
        coordination_mode="bootstrap",
        required_engine_type="sglang",
    )

    assert data["decode_workload_tokens"] == 13


def _resource_payload(role=PDRole.ROLE_P, instance_id=1):
    instance = Instance(id=instance_id, role=role, job_name="test", model_name="test")
    endpoint = Endpoint(id=0, ip="127.0.0.1", business_port="8000")
    return {
        "instance": instance.model_dump(mode="json"),
        "endpoint": endpoint.model_dump(mode="json"),
        "workload": {"active_tokens": 7},
    }


def _release_params(*, attempt_seq=1, role=PDRole.ROLE_P, instance_id=1):
    return UpdateWorkloadParams(
        instance_id=instance_id,
        endpoint_id=0,
        role=role,
        req_id="request-1",
        workload_action=WorkloadAction.RELEASE_TOKENS,
        workload_change=Workload(active_tokens=-7),
        attempt_seq=attempt_seq,
    )


async def _admit_single(client):
    client._transport.send_request = AsyncMock(
        side_effect=[
            _response({**_delivery(), **_resource_payload()}),
            _response(),
        ]
    )
    result, fallback = await client._select_and_allocate_tars_single(
        PDRole.ROLE_P,
        RequestInfo(req_id="request-1", req_data={}, req_len=1, api="/test"),
        attempt_seq=1,
        coordination_mode="trigger",
        required_engine_type=None,
    )
    assert result is not None and not fallback


async def test_agent_hint_bookkeeping_failure_does_not_orphan_confirmed_tars_allocation():
    client = _client()
    record = Mock(side_effect=RuntimeError("hint store unavailable"))
    client._session_affinity = SimpleNamespace(record=record)
    client._transport.send_request = AsyncMock(
        side_effect=[_response({**_delivery(), **_resource_payload()}), _response()]
    )
    req_info = RequestInfo(
        req_id="request-1",
        req_data={},
        req_len=1,
        api="/test",
        agent_hint_info=AgentHintInfo(session_id="agent-1"),
    )

    result = await client.select_and_allocate(PDRole.ROLE_P, req_info, attempt_seq=1)

    assert result is not None
    assert client.tars_has_attempt("request-1", 1)
    assert len(client._tars_workload_receipts) == 1
    record.assert_called_once_with("agent-1", 1, PDRole.ROLE_P.value, epoch=None, endpoint_id=0)


@pytest.mark.parametrize("event", ["request_complete", "abort"])
async def test_terminal_lifecycle_keeps_server_receipt_until_generic_cleanup(event):
    client = _client()
    await _admit_single(client)
    client._transport.send_request = AsyncMock(return_value=_response())

    assert await client._tars_lifecycle("request-1", 1, event)
    assert not client.tars_has_attempt("request-1", 1)
    # No native SHM is attached: both cleanup and duplicate cleanup must use the
    # server-owned receipt, even after the lifecycle removed the active attempt.
    assert await client.update_workload(_release_params())
    assert await client.update_workload(_release_params())
    requests = [call.args[0] for call in client._transport.send_request.call_args_list]
    assert [request.request_type for request in requests] == [
        SchedulerRequestType.TARS_LIFECYCLE,
        SchedulerRequestType.TARS_RELEASE_WORKLOAD,
    ]
    assert requests[-1].data == {
        "kind": "single",
        "req_id": "request-1",
        "attempt_seq": 1,
        "scheduler_epoch": 9,
        "delivery_token": "delivery-1",
        "instance_id": 1,
        "endpoint_id": 0,
        "role": PDRole.ROLE_P.value,
    }
    client.tars_forget_request("request-1")
    assert client._tars_workload_receipts == {}


async def test_ambiguous_server_release_keeps_receipt_and_retries_same_identity():
    client = _client()
    await _admit_single(client)
    client._transport.send_request = AsyncMock(side_effect=[None, _response()])
    assert not await client.update_workload(_release_params())
    client.tars_forget_request("request-1")
    assert len(client._tars_workload_receipts) == 1
    assert await client.update_workload(_release_params())
    first, second = [call.args[0] for call in client._transport.send_request.call_args_list]
    assert first.data == second.data


@pytest.mark.parametrize("ack", [{}, {"success": False}, {"success": 1}])
async def test_invalid_release_ack_does_not_forget_receipt_or_fall_back_to_cas(ack):
    client = _client()
    await _admit_single(client)
    client._transport.send_request = AsyncMock(return_value=_response(ack))
    assert not await client.update_workload(_release_params())
    client.tars_forget_request("request-1")
    assert len(client._tars_workload_receipts) == 1
    assert not next(iter(client._tars_workload_receipts.values()))["released"]


@pytest.mark.parametrize(
    ("attempt_seq", "role", "instance_id"),
    [
        (1, PDRole.ROLE_D, 2),  # Trigger D-first allocation belongs to the worker.
        (2, PDRole.ROLE_P, 1),  # A later fallback may reuse the same endpoint.
    ],
)
async def test_non_tars_receipt_allocations_keep_worker_cas(attempt_seq, role, instance_id):
    from motor.coordinator.scheduler.runtime.workload_shm.native import STATUS_OK

    client = _client()
    await _admit_single(client)
    native = SimpleNamespace(cas_sub_floor0=Mock(return_value=(STATUS_OK, 11)))
    client._workload_reader = SimpleNamespace(
        native=native,
        entry_meta=Mock(return_value={"generation": 4, "slot": 0}),
    )
    client._transport.send_request = AsyncMock()
    assert await client.update_workload(_release_params(attempt_seq=attempt_seq, role=role, instance_id=instance_id))
    native.cas_sub_floor0.assert_called_once_with(instance_id, 0, 4, 7.0, slot=0)
    client._transport.send_request.assert_not_called()


async def test_bootstrap_records_both_server_owned_resources():
    client = _client()
    payload = _delivery()
    payload.update(
        kind="joint",
        prefill=_resource_payload(),
        decode=_resource_payload(PDRole.ROLE_D, 2),
    )
    client._transport.send_request = AsyncMock(side_effect=[_response(payload), _response()])
    result = await client.tars_allocate_bootstrap_joint(
        RequestInfo(req_id="request-1", req_data={}, req_len=1, api="/test"),
        attempt_seq=1,
    )
    assert result is not None
    client._transport.send_request = AsyncMock(return_value=_response())
    assert await client.update_workload(_release_params())
    assert await client.update_workload(_release_params(role=PDRole.ROLE_D, instance_id=2))
    assert [call.args[0].request_type for call in client._transport.send_request.call_args_list] == [
        SchedulerRequestType.TARS_RELEASE_WORKLOAD,
        SchedulerRequestType.TARS_RELEASE_WORKLOAD,
    ]


async def test_reused_request_identity_cannot_overwrite_an_unacknowledged_receipt():
    client = _client()
    await _admit_single(client)
    client._transport.send_request = AsyncMock(return_value=_response())
    assert await client.tars_abort("request-1", 1)
    client.tars_forget_request("request-1")
    payload = {**_delivery(), **_resource_payload(), "delivery_token": "new-delivery"}
    client._transport.send_request = AsyncMock(side_effect=[_response(payload), _response()])

    result, fallback = await client._select_and_allocate_tars_single(
        PDRole.ROLE_P,
        RequestInfo(req_id="request-1", req_data={}, req_len=1, api="/test"),
        attempt_seq=1,
        coordination_mode="trigger",
        required_engine_type=None,
    )
    assert result is None and not fallback
    assert [call.args[0].request_type for call in client._transport.send_request.call_args_list] == [
        SchedulerRequestType.TARS_PREPARE_SINGLE,
        SchedulerRequestType.TARS_REVOKE,
    ]
    assert next(iter(client._tars_workload_receipts.values()))["delivery"]["delivery_token"] == "delivery-1"
