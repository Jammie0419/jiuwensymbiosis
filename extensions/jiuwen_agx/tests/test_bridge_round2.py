# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Round-2 fixes: partial-write safety, clear headless refusal, step guard."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_EXT = Path(__file__).resolve().parents[1]
_BRIDGE = _EXT / "scripts" / "agx_bridge_server.py"


def _load():
    spec = importlib.util.spec_from_file_location("agx_bridge_server_r2", _BRIDGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPartialWriteSafety:
    """大响应（如大 inventory）在非阻塞 socket 上分步排空，不再断连。"""

    def test_write_buffer_drains_via_send(self, monkeypatch):
        module = _load()
        adapter = module.AgxSceneAdapter(["swing"], {}, None, mode="viewer")

        sent = []

        class _Sock:
            def send(self, data):
                sent.append(data)
                return len(data)  # 全部送出

        adapter._conn = _Sock()
        adapter._write_buffer = b""
        adapter._send_response({"ok": True, "big": "x" * 5000})
        assert adapter._write_buffer == b""  # 全部排空
        assert sent and sum(len(x) for x in sent) == len(sent[0])  # 一次送出
        assert sent[0].startswith(b'{"ok": true')

    def test_partial_send_keeps_remainder_and_does_not_close(self, monkeypatch):
        module = _load()
        adapter = module.AgxSceneAdapter(["swing"], {}, None, mode="viewer")

        class _BusySock:
            def send(self, data):
                raise BlockingIOError  # 内核缓冲满

        adapter._conn = _BusySock()
        payload = b"x" * 100
        adapter._send_response_raw = None
        adapter._write_buffer = b""
        # 直接触发写入路径
        adapter._write_buffer += payload
        adapter._drain_writes()
        assert adapter._write_buffer == payload  # 原样保留
        assert adapter._conn is not None  # 没有断连


class TestHeadlessExcavatorRefusal:
    def test_clear_error_instead_of_environment_riddle(self):
        module = _load()
        adapter = module.AgxSceneAdapter(
            ["swing"], {}, None, mode="headless", build_excavator=True
        )
        with pytest.raises(ValueError, match="viewer"):
            adapter.load()


class TestStepClockGuard:
    def test_step_raises_when_clock_frozen(self, monkeypatch):
        module = _load()
        adapter = module.AgxSceneAdapter(["swing"], {}, None, mode="headless")

        class _FrozenSim:
            def getTimeStamp(self):
                return 0.0  # 时间永不前进

            def stepTo(self, t):
                pass

        adapter._sim = _FrozenSim()
        with pytest.raises(RuntimeError, match="not advancing"):
            adapter._step(1 / 60)
