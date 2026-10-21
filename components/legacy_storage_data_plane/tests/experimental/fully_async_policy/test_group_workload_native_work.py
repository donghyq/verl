"""Control response and matched-work safeguards; CPU fixtures only."""

import asyncio
import copy
import importlib.util
import io
import json
import pathlib
import sys
import urllib.error
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).parents[3]
filename = ROOT / "project-b-final-validation-20260928/terminal-20261007/gate2_tool_agent_runner.py"
spec = importlib.util.spec_from_file_location("gate_native_work_runner", filename)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def test_text_control_response_is_not_json_and_errors_keep_status_and_body(monkeypatch):
    class Response(io.BytesIO):
        status = 200

    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *args, **kwargs: Response(b"Cache flushed.\n"))
    receipt = runner.http_response("http://CPU-fixture/flush_cache")
    assert receipt["status"] == 200 and receipt["body"] == "Cache flushed.\n"
    with pytest.raises(json.JSONDecodeError):
        runner.http_json("http://CPU-fixture/generate")
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *args, **kwargs: Response(b'{"text":"real-json-shape"}'))
    assert runner.http_json("http://CPU-fixture/get_server_info") == {"text": "real-json-shape"}

    def rejected(*args, **kwargs):
        raise urllib.error.HTTPError("http://CPU-fixture/flush_cache", 400, "bad request", {}, io.BytesIO(b"Cache flushed."))

    monkeypatch.setattr(runner.urllib.request, "urlopen", rejected)
    assert runner.http_response("http://CPU-fixture/flush_cache")["status"] == 400
    with pytest.raises(RuntimeError, match="HTTP 400"):
        runner.http_json("http://CPU-fixture/generate")


@pytest.mark.parametrize("status,body,accepted", [
    (200, "Cache flushed.\nbackend details", True),
    (400, "Cache flushed.\nbackend details", False),
    (503, "busy", False),
    (200, '{"success": false}', False),
])
def test_window_reset_requires_real_success_and_saves_control_receipt(tmp_path, monkeypatch, status, body, accepted):
    instance = object.__new__(runner.ToolAgentRunner)
    instance.raw_dir = tmp_path
    instance.server_urls = {"a": "http://CPU-fixture-a", "b": "http://CPU-fixture-b"}
    monkeypatch.setattr(runner, "http_response", lambda url, *args: {"status": status, "body": body, "url": url})
    coroutine = instance.reset_window("sticky", 0)
    if accepted:
        asyncio.run(coroutine)
    else:
        with pytest.raises(RuntimeError, match="cache reset refused"):
            asyncio.run(coroutine)
    receipts = json.loads((tmp_path / "round-0-sticky-cache-reset.json").read_text())
    assert receipts["a"]["success"] is accepted
    assert receipts["a"]["attempts"][0]["body"] == body


def test_flush_method_fallback_is_only_for_405_and_preserves_both_receipts(tmp_path, monkeypatch):
    instance = object.__new__(runner.ToolAgentRunner)
    instance.raw_dir = tmp_path
    instance.server_urls = {"a": "http://CPU-fixture-a"}
    calls = []

    def response(url, body=None, method=None):
        calls.append(method or "GET")
        return {"status": 405 if method is None else 200, "body": "method rejected" if method is None else "Cache flushed."}

    monkeypatch.setattr(runner, "http_response", response)
    asyncio.run(instance.reset_window("least_inflight", 0))
    assert calls == ["GET", "POST"]
    receipts = json.loads((tmp_path / "round-0-least_inflight-cache-reset.json").read_text())
    assert [reply["status"] for reply in receipts["a"]["attempts"]] == [405, 200]


def test_flush_transport_failure_preserves_the_prior_method_receipt(tmp_path, monkeypatch):
    instance = object.__new__(runner.ToolAgentRunner)
    instance.raw_dir = tmp_path
    instance.server_urls = {"a": "http://CPU-fixture-a"}

    def response(url, body=None, method=None):
        if method == "POST":
            raise OSError("CPU fixture transport failure")
        return {"status": 405, "body": "method rejected"}

    monkeypatch.setattr(runner, "http_response", response)
    with pytest.raises(OSError, match="transport failure"):
        asyncio.run(instance.reset_window("sticky", 0))
    receipt = json.loads((tmp_path / "round-0-sticky-cache-reset.json").read_text())["a"]
    assert receipt["attempts"] == [{"status": 405, "body": "method rejected"}]
    assert not receipt["success"] and "transport failure" in receipt["error"]


def test_full_native_work_detects_tokens_after_output_clip_and_actual_tool_results():
    calls = [{
        "replica_id": "a", "start_at": 1.0, "end_at": 2.0,
        "request": {"input_ids": [1, 2], "sampling_params": {"sampling_seed": 42}},
        "result": {"text": "full output", "meta_info": {
            "output_token_logprobs": [[-0.1, 5], [-0.2, 6], [-0.3, 7]],
            "completion_tokens": 3, "finish_reason": {"type": "length"},
        }},
    }]
    tools = [{"name": "query_venue_detail", "parameters": {"venue_id": "V001"},
              "response": {"text": "original real tool result"}, "reward": 0.0, "error": None,
              "start_at": 1.0, "end_at": 2.0}]
    original = runner.native_work_record(calls, tools)
    assert original["observed_generated_tokens"] == 3
    changed = copy.deepcopy(calls)
    changed[0]["result"]["meta_info"]["output_token_logprobs"][-1][1] = 99
    assert runner.sha(runner.native_work_record(changed, tools)) != runner.sha(original)
    changed = copy.deepcopy(calls)
    changed[0]["request"]["sampling_params"]["sampling_seed"] = 43
    assert runner.sha(runner.native_work_record(changed, tools)) != runner.sha(original)
    changed = copy.deepcopy(calls)
    changed[0]["request"]["input_ids"].append(3)
    assert runner.sha(runner.native_work_record(changed, tools)) != runner.sha(original)
    changed_tools = copy.deepcopy(tools)
    changed_tools[0]["response"]["text"] = "different actual tool result"
    assert runner.sha(runner.native_work_record(calls, changed_tools)) != runner.sha(original)
    changed = copy.deepcopy(calls)
    changed[0].update(replica_id="b", start_at=3.0, end_at=4.0)
    changed[0]["result"]["meta_info"]["output_token_logprobs"][0][0] = -0.11
    assert runner.sha(runner.native_work_record(changed, tools)) == runner.sha(original)


def test_tool_recording_delegates_original_parameters_and_return_without_replay():
    parameters = {"venue_id": "V001"}
    agent_data = object()
    actual_calls = []
    response = SimpleNamespace(model_dump=lambda: {"text": "CPU fixture tool output"})
    actual_result = (response, 0.0, {})

    class Tool:
        name = "query_venue_detail"

        async def execute(self, instance_id, received, **kwargs):
            assert received is parameters and kwargs["agent_data"] is agent_data
            actual_calls.append(instance_id)
            return actual_result

    records = []
    result = asyncio.run(runner.RecordedTool(Tool(), records).execute("original-instance", parameters, agent_data=agent_data))
    assert result is actual_result and actual_calls == ["original-instance"]
    assert records[0]["parameters"] == parameters
    assert records[0]["response"] == {"text": "CPU fixture tool output"}
    assert records[0]["error"] is None
