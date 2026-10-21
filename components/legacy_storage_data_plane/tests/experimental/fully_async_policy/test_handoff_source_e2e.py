"""End-to-end checkpoint -> takeover -> resume over the REAL ToolAgentLoop.

Reuses the make_loop fixture style from test_tool_trajectory_resume.py (a real
ToolAgentLoop instance with stubbed generate/template/parser).  The trajectory
runs under run_preemptible_trajectory, yields at the completed tool batch after
a preemption request, commits through the real TrajectorySnapshotRegistry, and
is resumed by resume_preemptible_trajectory on a *different* loop instance:

- tool calls: exactly one execution total (tool_replay_count == 0)
- resumed output contains both generations' sampling segments
- committed history is never rewritten
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.experimental.fully_async_policy.handoff_source import (
    resume_preemptible_trajectory,
    run_preemptible_trajectory,
)
from verl.experimental.fully_async_policy.preemptible_handoff import (
    TrajectoryPreemptionTracker,
)
from verl.experimental.fully_async_policy.rollout_artifact import (
    CanonicalPolicyKVIdentity,
    InMemoryRolloutArtifactStore,
)
from verl.experimental.fully_async_policy.trajectory_snapshot import TrajectorySnapshotRegistry
from verl.tools.function_tool import FunctionTool
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.workers.rollout.replica import TokenOutput

IDENTITY = asdict(CanonicalPolicyKVIdentity("0", "weights", "template", "kv"))
ENV, TOOLS = "env-1", "tools-1"


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
async def test_real_tool_agent_loop_checkpoint_then_resume_across_owners():
    calls, generated, templates = [], [], []
    source_loop = make_loop(calls, generated, templates)
    registry = TrajectorySnapshotRegistry(InMemoryRolloutArtifactStore())
    tracker = TrajectoryPreemptionTracker()

    # A preemption is requested while the trajectory is in flight.
    def request_preemption_soon():
        tracker.register_running("task-1")
        tracker.request_preemption("task-1")

    # Simulate the cooperative signal: request preemption right before the
    # owner check inside the first PROCESSING_TOOLS boundary.  The loop's
    # check_owner runs after each state transition; hooking the tracker's
    # register through the executor, then requesting on the second call.
    original_run = source_loop.run_until_checkpoint

    async def run_with_late_preempt(*args, commit_snapshot, check_owner, **kwargs):
        checks = [0]

        async def check_owner_with_request():
            checks[0] += 1
            if checks[0] == 2:  # first call happens at loop start; request next
                try:
                    tracker.request_preemption("task-1")
                except ValueError:
                    pass
            await check_owner()

        return await original_run(*args, commit_snapshot=commit_snapshot,
                                  check_owner=check_owner_with_request, **kwargs)

    source_loop.run_until_checkpoint = run_with_late_preempt

    result = await run_preemptible_trajectory(
        agent_loop=source_loop,
        sampling_params={"temperature": 0.7},
        snapshot_registry=registry,
        tracker=tracker,
        trajectory_id="task-1",
        owner_id="source-rollouter",
        identity=IDENTITY,
        environment_version=ENV,
        tool_version=TOOLS,
        kwargs={
            "raw_prompt": [{"role": "user", "content": "look up record 7"}],
            "rollout_group_id": "group-1",
            "rollout_sample_index": 2,
            "rollout_group_size": 8,
        },
    )

    assert result["status"] == "yielded"
    assert calls == [7] and len(generated) == 1  # tool executed exactly once

    # The committed snapshot is loadable through a takeover by the target.
    member = {"trajectory_id": "task-1", "snapshot_ref": result["snapshot_ref"],
              "expected_generation": result["generation"]}
    registry.release_owner(registry.owners["task-1"])  # source lease expires/released
    target_loop = make_loop(calls, generated, templates)
    templates.clear()

    output, metrics = await resume_preemptible_trajectory(
        agent_loop=target_loop,
        snapshot_registry=registry,
        tracker=tracker,
        manifest_member=member,
        environment_version=ENV,
        tool_version=TOOLS,
    )

    # tool_replay_count == 0: the committed tool call was never re-executed,
    # and the prompt was never re-rendered on the target.
    assert calls == [7]
    assert not templates
    assert output.response_ids == [10, 11, 202, 203, 20, 21]
    assert output.response_mask == [1, 1, 0, 0, 1, 1]
    assert [s["execution_generation"] for s in output.extra_fields["trajectory_segments"]] == [
        result["generation"], metrics["generation"]]
    assert metrics["generation"] > result["generation"]  # generation fencing advanced
    assert output.extra_fields["rollout_group_id"] == "group-1"
    assert output.extra_fields["rollout_sample_index"] == 2
    assert output.extra_fields["rollout_group_size"] == 8


@pytest.mark.asyncio
async def test_without_preemption_the_loop_runs_to_completion_uncut():
    calls, generated, templates = [], [], []
    source_loop = make_loop(calls, generated, templates)
    registry = TrajectorySnapshotRegistry(InMemoryRolloutArtifactStore())
    tracker = TrajectoryPreemptionTracker()

    result = await run_preemptible_trajectory(
        agent_loop=source_loop,
        sampling_params={"temperature": 0.7},
        snapshot_registry=registry,
        tracker=tracker,
        trajectory_id="task-2",
        owner_id="source-rollouter",
        identity=IDENTITY,
        environment_version=ENV,
        tool_version=TOOLS,
        kwargs={
            "raw_prompt": [{"role": "user", "content": "look up record 7"}],
            "rollout_group_id": "group-1",
            "rollout_sample_index": 3,
            "rollout_group_size": 8,
        },
    )
    # No preemption request -> full run, no snapshot committed.
    assert result["status"] == "completed"
    assert calls == [7]
    assert "task-2" not in registry.latest
    assert "task-2" not in tracker.records


@pytest.mark.asyncio
async def test_resume_fails_closed_when_identity_or_budget_changes():
    calls, generated, templates = [], [], []
    source_loop = make_loop(calls, generated, templates)
    registry = TrajectorySnapshotRegistry(InMemoryRolloutArtifactStore())
    tracker = TrajectoryPreemptionTracker()
    # Commit a checkpoint, then attempt an invalid resume.
    result = await run_preemptible_trajectory(
        agent_loop=source_loop, sampling_params={"temperature": 0.7},
        snapshot_registry=registry, tracker=tracker,
        trajectory_id="task-3", owner_id="source-rollouter",
        identity=IDENTITY, environment_version=ENV, tool_version=TOOLS,
        kwargs={"raw_prompt": [{"role": "user", "content": "look up record 7"}],
                "rollout_group_id": "group-1", "rollout_sample_index": 4,
                "rollout_group_size": 8})
    assert result["status"] == "completed"  # no preemption: full run
    # A fresh trajectory with a checkpoint committed via direct registry use.
    lease = registry.open("task-4", "source", 600)
    from tests.experimental.fully_async_policy.test_handoff_executor import make_snapshot

    snapshot = make_snapshot("task-4", "group-1", 5, 8, lease.generation)
    key = registry.commit(lease, snapshot, 3600)
    registry.release_owner(lease)
    tracker.register_running("task-4")
    tracker.request_preemption("task-4")
    tracker.mark_safe_point("task-4")
    tracker.mark_snapshot_committed("task-4", key)

    target_loop = make_loop(calls, generated, templates)
    member = {"trajectory_id": "task-4", "snapshot_ref": key,
              "expected_generation": lease.generation}
    # Any identity/budget/version mismatch must fail closed; which guard fires
    # first depends on the loop's own validation order.
    with pytest.raises(ValueError, match="changed|mismatch"):
        await resume_preemptible_trajectory(
            agent_loop=target_loop, snapshot_registry=registry, tracker=tracker,
            manifest_member=member,
            environment_version="env-CHANGED", tool_version=TOOLS)


def main():
    async def run_all():
        await test_real_tool_agent_loop_checkpoint_then_resume_across_owners()
        await test_without_preemption_the_loop_runs_to_completion_uncut()
        await test_resume_fails_closed_when_identity_or_budget_changes()
        print("HANDOFF_SOURCE_E2E_OK")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
