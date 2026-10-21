"""CPU template/count regressions; no model output or engine request."""

import asyncio
import importlib.util
import json
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).parents[3]
FILE = ROOT / "project-b-final-validation-20260928/terminal-20261007/gate2_tool_agent_runner.py"
spec = importlib.util.spec_from_file_location("group_prompt_count_runner", FILE)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class TemplateFixture:
    """Only reproduce tokenizer output containers; never business tokens."""

    def __init__(self, shape):
        self.shape = shape
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        from transformers import BatchEncoding

        self.calls.append(kwargs)
        ids = [101, 102, 103, 104, 105, 106, 107]
        if self.shape == "flat":
            return ids
        if self.shape == "nested":
            return [ids]
        return BatchEncoding({"input_ids": [ids], "attention_mask": [[1] * len(ids)]})


def fixture_runner(shape):
    instance = object.__new__(runner.ToolAgentRunner)
    instance.tokenizer = TemplateFixture(shape)
    instance.schemas = [{"type": "function", "function": {"name": name}} for name in ("search", "detail")]
    instance.data_config = {"apply_chat_template_kwargs": {"enable_thinking": False}}
    return instance


async def actual_agent_loop_template(instance, member):
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase

    available = {schema["function"]["name"]: schema for schema in instance.schemas}
    selected = member.get("tool_selection")
    active = {name: available[name] for name in selected if name in available} if selected else available
    # Invoke the unchanged production method; bypass parser/model setup only.
    agent = SimpleNamespace(
        tokenizer=instance.tokenizer, processor=None, loop=asyncio.get_running_loop(),
        apply_chat_template_kwargs=instance.data_config["apply_chat_template_kwargs"],
    )
    return await AgentLoopBase.apply_chat_template(agent, member["raw_prompt"], tools=list(active.values()))


@pytest.mark.parametrize("shape", ["flat", "nested", "batch_encoding"])
def test_prepared_count_matches_actual_agent_loop_normalized_input(shape):
    instance = fixture_runner(shape)
    member = {"raw_prompt": [{"role": "user", "content": "CPU fixture only"}], "tool_selection": ["search", "detail"]}
    expected = asyncio.run(actual_agent_loop_template(instance, member))
    assert instance.prompt_token_count(member) == len(expected) == 7
    assert instance.tokenizer.calls[-1] == instance.tokenizer.calls[-2]
    assert instance.tokenizer.calls[-1]["return_dict"] is False


@pytest.mark.parametrize("selection,expected", [
    (["detail", "search", "detail", "unknown"], ["detail", "search"]),
    (["unknown"], []),
    ([], ["search", "detail"]),
])
def test_prepared_tools_keep_runtime_order_deduplication_and_unknown_selection(selection, expected):
    instance = fixture_runner("batch_encoding")
    member = {"raw_prompt": [{"role": "user", "content": "CPU fixture only"}], "tool_selection": selection}
    asyncio.run(actual_agent_loop_template(instance, member))
    runtime_tools = instance.tokenizer.calls[-1]["tools"]
    assert instance.prompt_token_count(member) == 7
    assert instance.tokenizer.calls[-1]["tools"] == runtime_tools
    assert [schema["function"]["name"] for schema in runtime_tools] == expected


def test_real_registered_tokenizer_lengths_and_strict_first_request_guard(tmp_path, monkeypatch):
    assets = os.environ.get("PROJECT_B_GATE2_TOKENIZER_ASSETS")
    if not assets:
        pytest.skip("requires the already collected private tokenizer assets; never download a model")
    from transformers import AutoTokenizer, BatchEncoding

    scripts = FILE.parent
    document = json.loads(pathlib.Path(os.environ.get("PROJECT_B_GATE2_WORKLOAD", scripts / "gate2_fixed_workload.json")).read_text())
    template = json.loads(pathlib.Path(os.environ.get(
        "PROJECT_B_GATE2_CALIBRATION_TEMPLATE", scripts / "gate2_independent_calibration_template.json")).read_text())
    document["engine_evidence_dir"] = str(tmp_path / "CPU-only")
    tokenizer = AutoTokenizer.from_pretrained(assets, local_files_only=True)
    instance = runner.ToolAgentRunner(document, {"a": "http://must-not-contact"}, tokenizer)
    members = template["calibration_members"] + [member for group in document["groups"] for member in group["members"]]
    assert len(members) == 36
    actual = {}
    for index, member in enumerate(members):
        ids = asyncio.run(actual_agent_loop_template(instance, member))
        actual[index] = ids
        assert instance.prompt_token_count(member) == len(ids)

    # Reproduce the measured container-length mistake with the actual assets.
    first = members[0]
    old = tokenizer.apply_chat_template(first["raw_prompt"], tools=instance.schemas, add_generation_prompt=True, tokenize=True)
    assert isinstance(old, BatchEncoding) and len(old) == 2 < len(actual[0])
    requests, releases = [], []

    class InterceptBeforeNetwork(Exception):
        pass

    def intercept(url, body, method=None):
        requests.append(body)
        raise InterceptBeforeNetwork("CPU fixture intercepts HTTP before network/model execution")

    monkeypatch.setattr(runner, "http_json", intercept)
    member = first

    def execute(prepared):
        context = SimpleNamespace(prompt_tokens=prepared, max_new_tokens=member["max_new_tokens"])
        return instance.execute(context=context, workload=member, acquire=lambda: "a",
                                release=lambda replica, generated=None: releases.append((replica, generated)))

    with pytest.raises(ValueError, match=rf"prepared=2 actual={len(actual[0])}"):
        asyncio.run(execute(2))
    assert requests == [] and releases == [("a", None)]
    with pytest.raises(InterceptBeforeNetwork):
        asyncio.run(execute(instance.prompt_token_count(member)))
    assert len(requests) == 1 and requests[0]["input_ids"] == actual[0]
    assert releases == [("a", None), ("a", None)]
