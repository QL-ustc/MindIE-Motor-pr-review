# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.

"""Bounded control-plane shutdown when TARS workload compensation fails."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from motor.config.coordinator import CoordinatorConfig
from motor.coordinator.scheduler.runtime import scheduler_server as server_module
from motor.coordinator.scheduler.runtime.scheduler_server import (
    AsyncSchedulerServer,
    _SchedulerRequestDispatcher,
)


@pytest.mark.parametrize("recovery", ["during_drain", "at_quiesce", "never"])
async def test_stop_bounds_failed_compensation_and_still_closes_resources(monkeypatch, recovery):
    config = CoordinatorConfig()
    config.scheduler_config.tars.drain_timeout_s = 0.1
    server = AsyncSchedulerServer(config, instance_manager=MagicMock())
    dispatcher = MagicMock(spec=_SchedulerRequestDispatcher)
    dispatcher._recovery_timers = {}
    server._dispatcher = dispatcher
    writer = MagicMock()
    publisher = MagicMock()
    transport = SimpleNamespace(disconnect=AsyncMock())
    context = MagicMock()
    server._workload_writer = writer
    server._pub_socket = publisher
    server._transport = transport
    server.context = context
    logger = MagicMock()
    monkeypatch.setattr(server_module, "logger", logger)

    now = 0.0
    quiesced = False
    delays = []

    async def advance(delay):
        nonlocal now
        delays.append(delay)
        now += delay

    def drain_complete():
        return (recovery == "during_drain" and now >= 0.05) or (recovery == "at_quiesce" and quiesced)

    def quiesce():
        nonlocal quiesced
        quiesced = True

    monkeypatch.setattr(server_module, "time", SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(server_module.asyncio, "sleep", advance)
    dispatcher.tars_drain_complete.side_effect = drain_complete
    dispatcher.quiesce_tars.side_effect = quiesce

    await server.stop()

    assert sum(delays) <= config.scheduler_config.tars.drain_timeout_s
    dispatcher.begin_tars_drain.assert_called_once()
    dispatcher.quiesce_tars.assert_called_once()
    writer.release.assert_called_once()
    publisher.close.assert_called_once()
    transport.disconnect.assert_awaited_once()
    context.term.assert_called_once()
    assert server._stop_event.is_set()
    assert server._workload_writer is None
    assert server.context is None
    if recovery == "during_drain":
        logger.warning.assert_not_called()
    else:
        logger.warning.assert_called_once()
    if recovery == "never":
        logger.error.assert_called_once()
        assert "forcing cold shutdown" in logger.error.call_args.args[0]
        assert "require recovery" in logger.error.call_args.args[0]
    else:
        logger.error.assert_not_called()
