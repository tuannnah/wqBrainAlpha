import json
from pathlib import Path

import pytest

from src.llm.agent_bridge import AgentBridgeClient


def test_complete_doc_response_co_san(tmp_path):
    # Có sẵn resp_001.json -> complete() đọc và trả về content
    (tmp_path / "resp_001.json").write_text(
        json.dumps({"content": '{"hypothesis": "x"}'}), encoding="utf-8"
    )
    client = AgentBridgeClient(str(tmp_path), poll_interval_s=0.01, timeout_s=2)
    out = client.complete("sys", "usr", json_mode=True, task="translate")

    assert out == '{"hypothesis": "x"}'
    # request được ghi ra để agent đọc
    req = json.loads((tmp_path / "req_001.json").read_text(encoding="utf-8"))
    assert req["system"] == "sys"
    assert req["user"] == "usr"
    assert req["task"] == "translate"
    # usage tăng (thô)
    assert client.usage.total_tokens > 0


def test_complete_timeout_khi_thieu_response(tmp_path):
    # clock giả nhảy vọt -> không cần chờ thật
    ticks = iter([0.0, 0.0, 100.0])
    client = AgentBridgeClient(
        str(tmp_path), timeout_s=10, poll_interval_s=0.01,
        clock=lambda: next(ticks), sleep=lambda s: None,
    )
    with pytest.raises(TimeoutError):
        client.complete("sys", "usr")
    # vẫn ghi request ra để agent có thể trả lời ở lần sau
    assert (tmp_path / "req_001.json").exists()


def test_complete_in_marker_stdout(tmp_path, capsys):
    (tmp_path / "resp_001.json").write_text(
        json.dumps({"content": "ok"}), encoding="utf-8"
    )
    client = AgentBridgeClient(str(tmp_path), poll_interval_s=0.01, timeout_s=2)
    client.complete("sys", "usr")
    out = capsys.readouterr().out
    assert "[[LLM_REQUEST 001]]" in out
