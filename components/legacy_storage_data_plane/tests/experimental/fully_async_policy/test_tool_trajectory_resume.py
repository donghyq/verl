from dataclasses import asdict
from types import SimpleNamespace

import pytest

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop, TrajectoryCheckpointYield
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.experimental.fully_async_policy.rollout_artifact import CanonicalPolicyKVIdentity
from verl.tools.function_tool import FunctionTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.workers.rollout.replica import TokenOutput


def make_loop(calls, generated, templates):
    loop = object.__new__(ToolAgentLoop)
    loop.processor = None
    loop.config = {"async_training": {}}
    loop.mm_processor_kwargs = {}
    loop.response_length = 100
    loop.max_user_turns = loop.max_assistant_turns = 8
    loop.max_parallel_calls = 1
    loop.max_tool_response_length = 10000
    loop.tool_response_truncate_side = "right"
    loop.tool_parser_name = "hermes"
    schema = OpenAIFunctionToolSchema.model_validate({"type": "function", "function": {
        "name": "read", "description": "Read an immutable record", "parameters": {
            "type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}}})
    def read(id):
        calls.append(id)
        return f"record {id}: verified"
    loop.tools = {"read": FunctionTool("read", read, schema)}
    loop.tool_schemas = [schema.model_dump()]
    async def render(messages, **kwargs):
        templates.append(messages)
        return [202, 203] if kwargs.get("remove_system_prompt") else [100, 101]
    loop.apply_chat_template = render
    async def generate(**kwargs):
        generated.append(list(kwargs["prompt_ids"]))
        tokens = [10, 11] if len(kwargs["prompt_ids"]) == 2 else [20, 21]
        return TokenOutput(token_ids=tokens, log_probs=[-0.1, -0.2], extra_fields={
            "trajectory_segments": [{"start": 0, "end": 2, "policy_version": "0",
                "sampling_config_fingerprint": "a" * 64}]})
    loop.server_manager = SimpleNamespace(generate=generate)
    async def parse(tokens, tools):
        return "", [FunctionCall(name="read", arguments='{"id":7}')] if tokens == [10, 11] else []
    loop.tool_parser = SimpleNamespace(extract_tool_calls=parse)
    return loop


@pytest.mark.asyncio
async def test_new_worker_continues_exact_tokens_without_replaying_committed_tool():
    calls, generated, templates = [], [], []
    a = make_loop(calls, generated, templates)
    committed, checks = [], []
    async def check():
        checks.append(True)
    async def commit(snapshot):
        committed.append(snapshot)
        return "shared/snapshot-1"
    with pytest.raises(TrajectoryCheckpointYield):
        await a.run_until_checkpoint({"temperature": 0.7}, commit_snapshot=commit, check_owner=check,
            raw_prompt=[{"role": "user", "content": "look up record 7"}],
            trajectory_id="task-1", rollout_group_id="group-1", rollout_sample_index=1,
            rollout_group_size=4, execution_generation=0, snapshot_environment_version="env-1",
            snapshot_tool_version="tools-1", snapshot_policy_identity=asdict(
                CanonicalPolicyKVIdentity("0", "weights", "template", "kv")))
    assert calls == [7] and len(generated) == 1
    snapshot = committed[0]
    assert snapshot.agent_state["prompt_ids"] == [100, 101, 10, 11, 202, 203]
    templates.clear()
    b = make_loop(calls, generated, templates)
    output = await b.resume_from_snapshot(snapshot, check_owner=check,
        execution_generation=1, environment_version="env-1", tool_version="tools-1")
    assert calls == [7] and not templates
    assert generated[-1] == snapshot.agent_state["prompt_ids"]
    assert output.prompt_ids == [100, 101]
    assert output.response_ids == [10, 11, 202, 203, 20, 21]
    assert output.response_mask == [1, 1, 0, 0, 1, 1]
    assert output.response_logprobs == [-0.1, -0.2, 0, 0, -0.1, -0.2]
    assert [(s["start"], s["end"]) for s in output.extra_fields["trajectory_segments"]] == [(0, 2), (2, 4)]
    assert len(checks) >= 5
    assert [s["execution_generation"] for s in output.extra_fields["trajectory_segments"]] == [0, 1]

    assert output.extra_fields["trajectory_id"] == "task-1"
    assert output.extra_fields["rollout_group_id"] == "group-1"
    assert output.extra_fields["rollout_sample_index"] == 1
    assert output.extra_fields["rollout_group_size"] == 4
    with pytest.raises(ValueError, match="identity/member"):
        await b.resume_from_snapshot(snapshot, check_owner=check, execution_generation=1,
            environment_version="env-1", tool_version="tools-1", rollout_sample_index=0)
