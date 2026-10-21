# Copyright 2026 Project B contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU contract regressions, with actual upstream CT and tool handlers.

Fake model results and a fixture tokenizer are not business/GPU evidence. No
model is built, and no training or distributed-storage throughput is measured.
"""

import asyncio
from collections import deque
from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.function_tool import FunctionTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.trainer.ppo.v1.trajectory_state import CanonicalPolicyKVIdentity
from verl.trainer.ppo.v1.trajectory_state.preemptible_tool_agent_loop import (
    DurableToolAgentLoop,
    TrajectoryCheckpointYield,
    _generation_segments,
)
from verl.utils.tokenizer.continuous_token import ContinuousTokenBuilder
from verl.workers.rollout.replica import TokenOutput


def identity(version=1):
    return asdict(CanonicalPolicyKVIdentity(str(version), "fixture-weights-" + str(version),
                                           "fixture-tokenizer-template", "fixture-kv-schema"))


def token_output(tokens=(31, 32), version=1, *, native=True):
    extra = {"global_steps": version, "min_global_steps": version, "max_global_steps": version}
    if native:
        extra["native_meta_info"] = {
            "weight_version": str(version),
            "weight_versions": [{"start": 0, "end": len(tokens), "version": str(version)}],
        }
    return TokenOutput(token_ids=list(tokens), log_probs=[-.5 - i / 10 for i in range(len(tokens))],
                       extra_fields=extra, stop_reason="completed")


class FixtureTokenizer:
    """Append-only text template so tests exercise the real CT suffix logic."""

    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, *, add_generation_prompt=False, **kwargs):
        self.calls += 1
        roles = {"system": 100, "user": 101, "assistant": 102, "tool": 103}
        tokens = []
        for message in messages:
            tokens.extend([roles[message["role"]], *[200 + ord(x) for x in message["content"]], 99])
        return tokens + ([104] if add_generation_prompt else [])


class FixtureParser:
    stop_token_ids = [99]

    async def extract_tool_calls(self, tokens, tools):
        if tokens and tokens[0] == 31:
            return "call search", [FunctionCall(name="search", arguments='{"q":"saved"}', tool_call_id="call-1")]
        return "final answer", []


class FixtureServer:
    def __init__(self, outputs):
        self.outputs = deque(outputs)
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return self.outputs.popleft()


def make_loop(outputs, tool_calls, *, tool_fn=None, response_length=4096):
    loop = DurableToolAgentLoop.__new__(DurableToolAgentLoop)
    loop.loop = asyncio.get_running_loop()
    loop.tokenizer = FixtureTokenizer()
    loop.continuous_token_builder = ContinuousTokenBuilder(loop.tokenizer, chat_template_kwargs={})
    loop.rollout_config = SimpleNamespace(full_determinism=False, prompt_length=4096)
    loop.prompt_length, loop.response_length = 4096, response_length
    loop.max_user_turns, loop.max_assistant_turns = 0, 0
    loop.max_parallel_calls, loop.max_tool_response_length = 2, 512
    loop.tool_response_truncate_side = "right"
    loop.tool_parser_name, loop.tool_parser = "fixture", FixtureParser()
    loop.processor = None
    loop.mm_processor_kwargs = {}
    if tool_fn is None:
        async def tool_fn(q):
            tool_calls.append(q)
            return "result " + q
    schema = OpenAIFunctionToolSchema.model_validate({"type": "function", "function": {
        "name": "search", "description": "fixture", "parameters": {
            "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"],
        },
    }})
    loop.tools = {"search": FunctionTool("search", tool_fn, schema, is_async=True)}
    loop.tool_schemas = [schema.model_dump(exclude_unset=True, exclude_none=True)]
    loop.server_manager = FixtureServer(outputs)
    return loop


def state_args(version=1):
    return {
        "trajectory_id": "g/member-0", "rollout_group_id": "g", "rollout_sample_index": 0,
        "rollout_group_size": 8, "execution_generation": 0,
        "snapshot_policy_identity": identity(version), "snapshot_environment_version": "env-v1",
        "snapshot_tool_version": "tool-v1", "raw_prompt": [{"role": "user", "content": "go"}],
    }


async def owner_ok():
    return None


async def checkpointed_source(*, tool_fn=None):
    calls = []
    loop = make_loop([token_output()], calls, tool_fn=tool_fn)
    saved = []

    async def commit(snapshot):
        saved.append(snapshot)
        return "inflight/fixture"

    with pytest.raises(TrajectoryCheckpointYield) as result:
        await loop.run_until_checkpoint({"temperature": 0, "logprobs": True}, commit_snapshot=commit,
                                        check_owner=owner_ok, **state_args())
    assert len(saved) == 1 and result.value.reference == "inflight/fixture"
    return loop, calls, saved[0]


def test_registry_and_opt_out_delegate_exactly(monkeypatch):
    assert _agent_loop_registry["durable_tool_agent"]["_target_"].endswith("DurableToolAgentLoop")
    sentinel = object()
    upstream = AsyncMock(return_value=sentinel)
    monkeypatch.setattr(ToolAgentLoop, "run", upstream)
    loop = DurableToolAgentLoop.__new__(DurableToolAgentLoop)
    assert asyncio.run(loop.run({"temperature": 1}, priority=7, raw_prompt=[{"content": "go"}])) is sentinel
    upstream.assert_awaited_once_with({"temperature": 1}, priority=7, raw_prompt=[{"content": "go"}])


def test_actual_ct_source_checkpoint_then_resume_preserves_history_and_never_replays_tool():
    async def scenario():
        _, calls, snapshot = await checkpointed_source()
        assert calls == ["saved"]
        assert snapshot.completed_tool_batches[0]["results"] == [
            {"role": "tool", "content": "result saved", "tool_call_id": "call-1"}]
        historical = deepcopy(snapshot.agent_state)
        target_calls = []
        target = make_loop([token_output((41, 42), version=2)], target_calls)
        target_server = target.server_manager
        result = await target.resume_from_snapshot(
            snapshot, check_owner=owner_ok, execution_generation=1, environment_version="env-v1",
            tool_version="tool-v1", snapshot_policy_identity=identity(2),
        )
        assert target_server.calls[0]["prompt_ids"] == historical["prompt_ids"]
        assert target.tokenizer.calls == 0  # No rendering of committed messages on target.
        assert target_calls == []
        assert result.prompt_ids + result.response_ids == historical["prompt_ids"] + [41, 42]
        assert result.response_mask[:-2] == historical["response_mask"]
        assert result.response_logprobs[:-2] == historical["response_logprobs"]
        assert snapshot.agent_state == historical  # Caller's committed object was not mutated.
        assert [(s["start"], s["end"], s["policy_version"], s["execution_generation"])
                for s in result.extra_fields["trajectory_segments"]] == [(0, 2, "1", 0), (2, 4, "2", 1)]
        assert result.extra_fields["completed_tool_batches"] == snapshot.completed_tool_batches
        receipts = result.extra_fields["trajectory_generation_receipts"]
        assert [r["extra_fields"]["native_meta_info"]["weight_version"] for r in receipts] == ["1", "2"]
    asyncio.run(scenario())


def test_checkpoint_waits_until_running_tool_batch_and_owner_check_finish():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        committed = []

        async def slow_tool(q):
            entered.set()
            await release.wait()
            return "done " + q

        async def commit(snapshot):
            committed.append(snapshot)
            return "inflight/fixture"

        loop = make_loop([token_output()], [], tool_fn=slow_tool)
        task = asyncio.create_task(loop.run_until_checkpoint(
            {"temperature": 0, "logprobs": True}, commit_snapshot=commit,
            check_owner=owner_ok, **state_args()))
        await entered.wait()
        assert committed == [] and not task.done()
        release.set()
        with pytest.raises(TrajectoryCheckpointYield):
            await task
        assert committed[0].completed_tool_batches[0]["results"][0]["content"] == "done saved"
    asyncio.run(scenario())


def test_functional_boundary_waits_after_real_tool_and_before_snapshot():
    async def scenario():
        reached, release = asyncio.Event(), asyncio.Event()
        calls, saved, observed = [], [], []

        async def boundary(turn_index, batch_count):
            observed.append((turn_index, batch_count))
            reached.set()
            await release.wait()

        async def commit(snapshot):
            saved.append(snapshot)
            return "inflight/fixture"

        loop = make_loop([token_output()], calls)
        task = asyncio.create_task(loop.run_until_checkpoint(
            {"temperature": 0, "logprobs": True}, commit_snapshot=commit,
            check_owner=owner_ok, _trajectory_acceptance_tool_boundary=boundary,
            **state_args()))
        await asyncio.wait_for(reached.wait(), timeout=5)
        assert calls == ["saved"]
        assert observed == [(0, 1)] and saved == [] and not task.done()
        release.set()
        with pytest.raises(TrajectoryCheckpointYield):
            await task
        assert saved[0].completed_tool_batches[0]["results"][0]["content"] == "result saved"
    asyncio.run(scenario())


def test_invalid_functional_boundary_rejected_before_generation_or_tool():
    async def scenario():
        calls, saved = [], []
        loop = make_loop([token_output()], calls)
        async def commit(snapshot):
            saved.append(snapshot)
        with pytest.raises(ValueError, match="boundary callback"):
            await loop.run_until_checkpoint(
                {"temperature": 0, "logprobs": True}, commit_snapshot=commit,
                check_owner=owner_ok, _trajectory_acceptance_tool_boundary=42,
                **state_args())
        assert calls == [] and saved == [] and loop.server_manager.calls == []
    asyncio.run(scenario())


def test_owner_expiring_during_tool_batch_prevents_snapshot_commit():
    async def scenario():
        checked, saved = [], []

        async def owner():
            checked.append(1)
            if len(checked) == 4:
                raise ValueError("fixture stale owner")

        async def commit(snapshot):
            saved.append(snapshot)
            return "inflight/fixture"

        loop = make_loop([token_output()], [])
        with pytest.raises(ValueError, match="stale owner"):
            await loop.run_until_checkpoint({"logprobs": True}, commit_snapshot=commit,
                                            check_owner=owner, **state_args())
        assert saved == []
    asyncio.run(scenario())


def test_callback_declining_preemption_continues_in_place():
    async def scenario():
        calls, saved = [], []
        loop = make_loop([token_output(), token_output((41,))], calls)

        async def commit(snapshot):
            saved.append(snapshot)
            return None

        result = await loop.run_until_checkpoint({"logprobs": True}, commit_snapshot=commit,
                                                check_owner=owner_ok, **state_args())
        assert calls == ["saved"] and len(saved) == 1
        assert result.response_ids[-1] == 41
        assert result.extra_fields["trajectory_segments"][-1]["start"] == 2
    asyncio.run(scenario())


@pytest.mark.parametrize("change,match", [
    ("generation", "must advance"), ("environment", "tool/environment"),
    ("tool", "tool/environment"), ("template", "tokenizer/template"),
    ("builder", "Continuous Token"), ("schema", "tool schema"),
    ("limits", "budgets/sampling"), ("member", "identity/member"),
    ("missing_ct", "Continuous Token"), ("selection", "tool selection"),
])
def test_resume_rejects_changed_contract_before_generation(change, match):
    async def scenario():
        _, _, snapshot = await checkpointed_source()
        target = make_loop([token_output((41,))], [])
        server = target.server_manager
        kwargs = {"check_owner": owner_ok, "execution_generation": 1, "environment_version": "env-v1",
                  "tool_version": "tool-v1", "snapshot_policy_identity": identity()}
        if change == "generation":
            kwargs["execution_generation"] = 0
        elif change == "environment":
            kwargs["environment_version"] = "env-v2"
        elif change == "tool":
            kwargs["tool_version"] = "tool-v2"
        elif change == "template":
            kwargs["snapshot_policy_identity"]["tokenizer_template_fingerprint"] = "other-template"
        elif change == "builder":
            target.continuous_token_builder.chat_template_kwargs["changed"] = True
        elif change == "schema":
            target.tools["search"].tool_schema.function.description = "different schema"
        elif change == "limits":
            target.response_length += 1
        elif change == "member":
            kwargs["rollout_sample_index"] = 1
        elif change == "missing_ct":
            snapshot.agent_state["extra_fields"].pop("trajectory_ct_identity")
        elif change == "selection":
            kwargs["extra_info"] = {"tool_selection": []}
        with pytest.raises(ValueError, match=match):
            await target.resume_from_snapshot(snapshot, **kwargs)
        assert server.calls == []
    asyncio.run(scenario())


def test_multi_version_generate_cannot_be_stamped_with_its_final_policy():
    output = token_output()
    output.extra_fields["native_gateway_attempts"] = [
        {"output_start": 0, "output_end": 1, "native_meta_info": {
            "weight_version": "1", "weight_versions": [{"start": 0, "end": 1, "version": "1"}]}},
        {"output_start": 1, "output_end": 2, "native_meta_info": {
            "weight_version": "2", "weight_versions": [{"start": 0, "end": 1, "version": "2"}]}},
    ]
    output.extra_fields["max_global_steps"] = 2
    with pytest.raises(ValueError, match="canonical policy identity"):
        _generation_segments(output, {}, identity(2))


def test_legacy_server_single_version_receipt_is_accepted_without_inventing_native_receipt():
    output = token_output(native=False)
    segments = _generation_segments(output, {"logprobs": True}, identity())
    assert [(row["start"], row["end"], row["policy_version"]) for row in segments] == [(0, 2, "1")]
    assert "native_meta_info" not in output.extra_fields


def test_upstream_output_cap_keeps_only_actual_retained_policy_spans():
    async def scenario():
        loop = make_loop([token_output((41, 42, 43))], [], response_length=2)
        result = await loop.run({"logprobs": True}, _trajectory_check_owner=owner_ok, **state_args())
        assert result.response_ids == [41, 42]
        assert result.response_logprobs == [-.5, -.6]
        assert [(row["start"], row["end"]) for row in result.extra_fields["trajectory_segments"]] == [(0, 2)]
        # The complete observed response remains in the receipt; it is not used
        # to claim that the capped training response contains all three tokens.
        assert result.extra_fields["trajectory_generation_receipts"][0]["generated_end"] == 3
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["logs", "no_version", "gap", "mixed_tp", "routed"])
def test_incomplete_actual_provenance_fails_closed(kind):
    output = token_output()
    if kind == "logs":
        output.log_probs = None
    elif kind == "no_version":
        output.extra_fields = {}
    elif kind == "gap":
        output.extra_fields["native_gateway_attempts"] = [{
            **output.extra_fields, "output_start": 1, "output_end": 2}]
    elif kind == "mixed_tp":
        output.extra_fields["native_meta_info"]["weight_versions"] = [
            {"start": 0, "end": 1, "version": "1"}, {"start": 1, "end": 2, "version": "2"}]
    elif kind == "routed":
        output.routed_experts = [[1]]
    with pytest.raises(ValueError):
        _generation_segments(output, {}, identity())


def test_ct_cannot_drop_old_generated_tokens_at_tool_boundary(monkeypatch):
    async def scenario():
        original = ContinuousTokenBuilder.merge_context_tokens

        def corrupt(builder, previous_messages, updated_messages, runtime_token_ids, **kwargs):
            result = original(builder, previous_messages, updated_messages, runtime_token_ids, **kwargs)
            return replace(result, token_ids=runtime_token_ids[:-1] + result.token_ids[len(runtime_token_ids):],
                           removed_prefix_token_count=1)

        monkeypatch.setattr(ContinuousTokenBuilder, "merge_context_tokens", corrupt)
        loop, saved = make_loop([token_output()], []), []

        async def commit(snapshot):
            saved.append(snapshot)
            return "inflight/fixture"

        with pytest.raises(ValueError, match="historical generated tokens"):
            await loop.run_until_checkpoint({"logprobs": True}, commit_snapshot=commit,
                                            check_owner=owner_ok, **state_args())
        assert saved == []
    asyncio.run(scenario())


def test_unreviewed_builder_and_multimodal_state_rejected():
    async def scenario():
        loop = make_loop([token_output()], [])

        class CustomBuilder(ContinuousTokenBuilder):
            pass

        loop.continuous_token_builder = CustomBuilder(loop.tokenizer)
        with pytest.raises(ValueError, match="audited upstream text"):
            await loop.run_until_checkpoint({}, commit_snapshot=owner_ok, check_owner=owner_ok, **state_args())
        loop.continuous_token_builder = ContinuousTokenBuilder(loop.tokenizer)
        kwargs = state_args()
        kwargs["raw_prompt"] = [{"role": "user", "content": [{"type": "image", "image": "fixture"}]}]
        with pytest.raises(ValueError, match="text-only"):
            await loop.run_until_checkpoint({}, commit_snapshot=owner_ok, check_owner=owner_ok, **kwargs)
        assert loop.server_manager.calls == []
    asyncio.run(scenario())


def test_concurrent_calls_keep_generation_receipts_separate():
    async def scenario():
        loop = make_loop([], [])

        class Server:
            async def generate(self, **kwargs):
                await asyncio.sleep(0)
                return token_output((41 if kwargs["prompt_ids"][1] == 311 else 42,))

        loop.server_manager = Server()
        first, second = state_args(), state_args()
        first["raw_prompt"] = [{"role": "user", "content": "o"}]
        second["raw_prompt"] = [{"role": "user", "content": "p"}]
        second["trajectory_id"], second["rollout_sample_index"] = "g/member-1", 1
        results = await asyncio.gather(*[
            loop.run({"logprobs": True}, _trajectory_check_owner=owner_ok, **kwargs)
            for kwargs in (first, second)])
        assert [result.response_ids for result in results] == [[41], [42]]
        assert [len(result.extra_fields["trajectory_generation_receipts"]) for result in results] == [1, 1]
        assert [result.extra_fields["rollout_sample_index"] for result in results] == [0, 1]
    asyncio.run(scenario())
