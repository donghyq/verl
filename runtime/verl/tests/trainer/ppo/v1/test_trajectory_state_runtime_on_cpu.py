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

"""Mock AgentLoop callbacks with the real durable coordinator; no model/TQ."""

import asyncio
from dataclasses import asdict

import pytest

from verl.trainer.ppo.v1.trajectory_state import (
    CanonicalPolicyKVIdentity,
    DurableTrajectoryCoordinator,
    HandoffManifest,
    encode_handoff_member_record,
)
from verl.trainer.ppo.v1.trajectory_state.preemptible_tool_agent_loop import TrajectoryCheckpointYield
from verl.trainer.ppo.v1.trajectory_state.runtime import PromptGroupExecutionError, PromptGroupHandoffRuntime, _Member
from verl.trainer.ppo.v1.trajectory_state.snapshot import TrajectorySnapshot


IDENTITY = asdict(CanonicalPolicyKVIdentity("1", "fixture-weight", "fixture-tokenizer", "fixture-kv"))


class AsyncClient:
    """Only adapts sync calls; coordinator validation/storage are not mocked."""

    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.calls = []

    def __getattr__(self, name):
        async def invoke(*args, **kwargs):
            self.calls.append(name)
            return getattr(self.coordinator, name)(*args, **kwargs)
        return invoke


def make_runtime(coordinator, *, owner="source", **kwargs):
    return PromptGroupHandoffRuntime(AsyncClient(coordinator), owner_id=owner, incarnation_id=owner + "-inc-0",
                                    policy_identity=IDENTITY, environment_version="env-v1", tool_version="tool-v1",
                                    **kwargs)


def input_kwargs(n):
    return [{"sampling_params": {"temperature": .7, "logprobs": True},
             "trajectory": {"original_index": i, "trace": "fixture"}, "agent_name": "durable_tool_agent",
             "raw_prompt": [{"role": "user", "content": f"member {i}"}],
             "original_tuple": (i, i + 1)} for i in range(n)]


def snapshot(uid, index, n, lease, kwargs):
    return TrajectorySnapshot(
        trajectory_id=lease.trajectory_id, group_id=uid, member_index=index, group_size=n,
        execution_generation=lease.generation,
        agent_state={"messages": [{"role": "user", "content": "query"}, {"role": "assistant", "content": "search"},
                                  {"role": "tool", "content": "saved result", "tool_call_id": "fixture-call"}],
                     "prompt_ids": [11, 12, 31, 40, 41], "response_ids": [31], "response_mask": [1, 0, 0],
                     "response_logprobs": [-.5, 0., 0.], "turn_scores": [], "tool_rewards": [0.],
                     "user_turns": 1, "assistant_turns": 1,
                     "extra_fields": {"trajectory_segments": [{"start": 0, "end": 1, "policy_version": "1",
                         "sampling_config_fingerprint": "a" * 64, "turn_index": 0, "execution_generation": lease.generation}]},
                     "tools_kwargs": {}, "metrics": {}, "request_id": "fixture-request", "mm_processor_kwargs": {}},
        sampling_params=kwargs["sampling_params"], limits={}, environment_version="env-v1", tool_version="tool-v1",
        policy_identity=IDENTITY, tool_selection=["search"],
        completed_tool_batches=[{"turn_index": 0, "calls": [{"name": "search", "arguments": "{}"}],
                                 "results": [{"role": "tool", "content": "saved result"}]}],
    )


async def wait_for(predicate):
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(.001)
    raise AssertionError("fixture condition did not arrive")


async def mixed_source(runtime, *, uid="g"):
    release = asyncio.Event()
    invoked = []

    async def run_member(index, kwargs, lease, commit_snapshot, check_owner):
        invoked.append(index)
        assert kwargs["trajectory"]["original_index"] == index
        await check_owner()
        if index < 2:
            return {"member_index": index, "origin": "source", "token_ids": [index + 10]}
        await release.wait()
        value = snapshot(uid, index, 8, lease, kwargs)
        reference = await commit_snapshot(value)
        assert reference is not None
        raise TrajectoryCheckpointYield(value, reference)

    task = asyncio.create_task(runtime.run_source(uid, input_kwargs(8), run_member, start_members=list(range(6))))
    await wait_for(lambda: runtime.group_status(uid) is not None
                   and runtime.group_status(uid)["completed_members"] == [0, 1]
                   and runtime.group_status(uid)["running_members"] == [2, 3, 4, 5])
    status = await runtime.request_preemption(uid)
    assert status["preemption_requested"]
    release.set()
    return await task, invoked


def test_mixed_n8_manifest_releases_then_target_returns_exactly_one_ordered_output(tmp_path):
    async def scenario():
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            source = make_runtime(coordinator)
            result, source_calls = await mixed_source(source)
            assert result.status == "handed_off" and result.outputs == []
            assert "/" not in result.handoff_id and ".." not in result.handoff_id
            assert source_calls == list(range(6))
            manifest = coordinator.get_handoff_manifest(result.handoff_id)
            assert [m["member_index"] for m in manifest.completed_members] == [0, 1]
            assert [m["member_index"] for m in manifest.checkpointed_trajectories] == [2, 3, 4, 5]
            assert [m["member_index"] for m in manifest.pending_samples] == [6, 7]
            assert coordinator.summary()["active_execution_leases"] == 0
            assert source.group_status("g")["handoff_id"] == result.handoff_id
            with pytest.raises(ValueError, match="already-active"):
                await source.run_source("g", input_kwargs(8), lambda *args: None)
            target = make_runtime(coordinator, owner="target")
            resumed, pending = [], []

            async def resume(index, saved, kwargs, lease, check_owner):
                resumed.append(index)
                assert saved.agent_state["prompt_ids"] == [11, 12, 31, 40, 41]
                assert lease.generation > saved.execution_generation
                assert kwargs == input_kwargs(8)[index]
                await check_owner()
                return {"member_index": index, "origin": "target-resume", "token_ids": [index + 10]}

            async def first_run(index, kwargs, lease, check_owner):
                pending.append(index)
                assert lease.generation == 0 and kwargs == input_kwargs(8)[index]
                await check_owner()
                return {"member_index": index, "origin": "target-pending", "token_ids": [index + 10]}

            completed = await target.run_target(result.handoff_id, resume, first_run)
            assert completed.status == "completed"
            assert [x["member_index"] for x in completed.outputs] == list(range(8))
            assert completed.member_kwargs == input_kwargs(8)
            assert resumed == [2, 3, 4, 5] and pending == [6, 7]
            assert coordinator.summary()["active_execution_leases"] == 0
            second = await target.run_target(result.handoff_id, resume, first_run)
            assert second.outputs == completed.outputs
            assert resumed == [2, 3, 4, 5] and pending == [6, 7]  # No duplicate callback.
            assert result.metrics["worker_reclaim_s"] is None
            assert result.metrics["source_execution_release_s"] >= 0
        finally:
            coordinator.close()
    asyncio.run(scenario())


def test_normal_complete_group_uses_no_manifest_or_queue_and_does_not_change_kwargs(tmp_path):
    async def scenario():
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            runtime = make_runtime(coordinator)
            called = []
            originals = input_kwargs(3)

            async def run_member(index, kwargs, lease, commit, check):
                called.append(index)
                assert kwargs == originals[index]
                await check()
                kwargs["raw_prompt"][0]["content"] = "callback-private-mutation"
                return (index, lease.generation)

            result = await runtime.run_source("normal", originals, run_member, start_members=[0])
            assert result.status == "completed" and result.handoff_id is None
            assert result.outputs == [(0, 0), (1, 0), (2, 0)]
            assert called == [0, 1, 2] and result.member_kwargs == originals == input_kwargs(3)
            assert "prepare_handoff_manifest" not in runtime.client.calls
            assert "prepare_publication" not in runtime.client.calls
            assert len(result.output_record_refs) == 3
            assert runtime.group_status("absent") is None
            assert await runtime.request_preemption("absent") is False
        finally:
            coordinator.close()
    asyncio.run(scenario())


def test_deadline_fences_live_execution_without_claiming_group_handoff(tmp_path):
    async def scenario():
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            runtime = make_runtime(coordinator, preemption_deadline_s=.025)
            never = asyncio.Event()
            saved_leases = []

            async def run_member(index, kwargs, lease, commit, check):
                saved_leases.append(lease)
                if index == 0:
                    return {"completed": 0}
                await never.wait()

            task = asyncio.create_task(runtime.run_source("timeout", input_kwargs(2), run_member))
            await wait_for(lambda: runtime.group_status("timeout") is not None
                           and runtime.group_status("timeout")["completed_members"] == [0]
                           and runtime.group_status("timeout")["running_members"] == [1])
            await runtime.request_preemption("timeout")
            with pytest.raises(PromptGroupExecutionError) as failed:
                await task
            assert failed.value.reason == "SAFE_POINT_TIMEOUT"
            assert failed.value.handoff_id is None and failed.value.unsettled == []
            assert coordinator.get_trajectory_preemption("timeout/member-1") == "aborted"
            assert coordinator.summary()["active_execution_leases"] == 0
            for lease in saved_leases:
                with pytest.raises(ValueError, match="stale"):
                    coordinator.check_trajectory_owner(lease)
            assert "prepare_handoff_manifest" not in runtime.client.calls
        finally:
            coordinator.close()
    asyncio.run(scenario())


def test_target_restart_retains_already_committed_result_and_reacquires_other_generations(tmp_path):
    async def scenario():
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        source = make_runtime(coordinator)
        result, _ = await mixed_source(source)
        calls = []
        blocked = asyncio.Event()
        target = make_runtime(coordinator, owner="target-0")

        async def first_resume(index, saved, kwargs, lease, check):
            calls.append(index)
            if index == 2:
                return {"member_index": index, "preserved": True}
            if index == 3:
                raise RuntimeError("fixture target process failure")
            await blocked.wait()

        async def first_pending(index, kwargs, lease, check):
            await blocked.wait()

        with pytest.raises(PromptGroupExecutionError, match="TARGET_EXECUTION_FAILED"):
            await target.run_target(result.handoff_id, first_resume, first_pending)
        assert set(coordinator.get_handoff_restored_members(result.handoff_id)) == {2}
        coordinator.close()
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            replacement = make_runtime(coordinator, owner="target-1")
            resumed = []

            async def second_resume(index, saved, kwargs, lease, check):
                resumed.append(index)
                assert index != 2
                await check()
                return {"member_index": index}

            async def second_pending(index, kwargs, lease, check):
                await check()
                return {"member_index": index}

            restored = await replacement.run_target(result.handoff_id, second_resume, second_pending)
            assert restored.outputs[2] == {"member_index": 2, "preserved": True}
            assert resumed == [3, 4, 5]
            assert sorted(restored.output_record_refs) == list(range(8))
            assert coordinator.summary()["active_execution_leases"] == 0
        finally:
            coordinator.close()
    asyncio.run(scenario())


def test_target_identity_mismatch_and_fabricated_checkpoint_fail_closed(tmp_path):
    async def scenario():
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            source = make_runtime(coordinator)
            result, _ = await mixed_source(source)
            target = make_runtime(coordinator, owner="target")
            target.policy_identity["model_weights_fingerprint"] = "different"
            callback = lambda *args: None
            with pytest.raises(PromptGroupExecutionError) as error:
                await target.run_target(result.handoff_id, callback, callback)
            assert isinstance(error.value.__cause__, ValueError)
            assert coordinator.get_handoff_restored_members(result.handoff_id) == {}
            bad_source = make_runtime(coordinator, owner="bad-source")

            async def fabricated(index, kwargs, lease, commit, check):
                raise TrajectoryCheckpointYield(snapshot("bad", index, 1, lease, kwargs), "unknown-ref")

            with pytest.raises(PromptGroupExecutionError) as fabricated_error:
                await bad_source.run_source("bad", input_kwargs(1), fabricated)
            assert "durable snapshot commit" in str(fabricated_error.value.__cause__)
            assert coordinator.summary()["active_execution_leases"] == 0
        finally:
            coordinator.close()
    asyncio.run(scenario())


def test_old_source_cleanup_valid_read_then_restart_takeover_cannot_abort_new_generation(tmp_path):
    async def scenario():
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            generation = coordinator.register_producer("source", "source-inc-0")
            source = coordinator.open_trajectory("g/member-0", "source", 600)
            coordinator.register_running_trajectory(source.trajectory_id)
            coordinator.request_trajectory_preemption(source.trajectory_id)
            coordinator.mark_trajectory_safe_point(source.trajectory_id)
            snapshot_ref = coordinator.commit_trajectory_snapshot(source, snapshot("g", 0, 2, source, input_kwargs(2)[0]), 900)
            old_pending = coordinator.open_trajectory("g/member-1", "source", 600)
            coordinator.register_running_trajectory(old_pending.trajectory_id)
            input0 = coordinator.put_handoff_member_record(encode_handoff_member_record(input_kwargs(2)[0]))
            input1 = coordinator.put_handoff_member_record(encode_handoff_member_record(input_kwargs(2)[1]))
            base0 = {"trajectory_id": "g/member-0", "group_id": "g", "member_index": 0, "group_size": 2}
            base1 = {"trajectory_id": "g/member-1", "group_id": "g", "member_index": 1, "group_size": 2}
            manifest = HandoffManifest("handoff", "source", generation, IDENTITY, "env-v1", "tool-v1", (),
                                       ({**base0, "snapshot_ref": snapshot_ref, "expected_generation": 0, "input_ref": input0},),
                                       (), ({**base1, "record_ref": input1},))
            coordinator.prepare_handoff_manifest(manifest)
            coordinator.commit_handoff_manifest("handoff")
            coordinator.release_handoff_source("handoff")
            # The old cleanup's separate read could legitimately have passed.
            coordinator.check_trajectory_owner(old_pending)
            # A restart invalidates the old execution lease; target acquisition
            # then advances the same existing generation history before abort.
            coordinator.close()
            coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
            coordinator.begin_handoff_restore("handoff")
            target = coordinator.acquire_handoff_pending("handoff", "g/member-1", "target", 600)
            assert target.generation == old_pending.generation + 1
            with pytest.raises(ValueError, match="stale"):
                coordinator.abort_owned_trajectory(old_pending)
            runtime = make_runtime(coordinator)
            stale = _Member(1, old_pending.trajectory_id, input_kwargs(2)[1], status="running",
                            lease=old_pending, tracker_registered=True)
            assert await runtime._fence_and_cancel([stale]) == []
            coordinator.check_trajectory_owner(target)
            assert coordinator.get_trajectory_preemption("g/member-1") == "running"
            assert "mark_trajectory_aborted" not in runtime.client.calls
            assert "check_trajectory_owner" not in runtime.client.calls
            assert runtime.client.calls == ["abort_owned_trajectory"]
        finally:
            coordinator.close()
    asyncio.run(scenario())


def test_atomic_owned_abort_releases_current_source_and_persists_tracker(tmp_path):
    coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
    try:
        owner = coordinator.open_trajectory("g/member-0", "source", 600)
        coordinator.register_running_trajectory(owner.trajectory_id)
        coordinator.request_trajectory_preemption(owner.trajectory_id)
        result = coordinator.abort_owned_trajectory(owner)
        assert result == {"trajectory_id": owner.trajectory_id, "state": "aborted", "released": True}
        assert coordinator.summary()["active_execution_leases"] == 0
        with pytest.raises(ValueError, match="stale"):
            coordinator.check_trajectory_owner(owner)
        coordinator.close()
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        assert coordinator.get_trajectory_preemption(owner.trajectory_id) == "aborted"
        assert coordinator.summary()["active_execution_leases"] == 0
    finally:
        coordinator.close()


def test_atomic_owned_abort_rejects_expiry_and_preserves_committed_snapshot(tmp_path):
    clock = [10.]
    coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run", clock=lambda: clock[0])
    try:
        expired = coordinator.open_trajectory("expired/member-0", "source", 1)
        coordinator.register_running_trajectory(expired.trajectory_id)
        coordinator.check_trajectory_owner(expired)  # Old read succeeds before time advances.
        clock[0] = 11.
        with pytest.raises(ValueError, match="stale"):
            coordinator.abort_owned_trajectory(expired)
        assert coordinator.get_trajectory_preemption(expired.trajectory_id) == "running"
        owner = coordinator.open_trajectory("g/member-0", "source", 600)
        coordinator.register_running_trajectory(owner.trajectory_id)
        coordinator.request_trajectory_preemption(owner.trajectory_id)
        coordinator.mark_trajectory_safe_point(owner.trajectory_id)
        committed = snapshot("g", 0, 1, owner, input_kwargs(1)[0])
        reference = coordinator.commit_trajectory_snapshot(owner, committed, 900)
        result = coordinator.abort_owned_trajectory(owner)
        assert result == {"trajectory_id": owner.trajectory_id, "state": "snapshot_committed", "released": False}
        coordinator.check_trajectory_owner(owner)
        assert coordinator.load_trajectory_snapshot(owner).encode() == committed.encode()
        assert coordinator.get_trajectory_preemption(owner.trajectory_id) == "snapshot_committed"
        assert reference is not None
    finally:
        coordinator.close()


def test_owned_control_valid_read_then_restart_rejects_old_generation_and_accepts_new(tmp_path):
    coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
    try:
        producer = coordinator.register_producer("source", "source-inc-0")
        source = coordinator.open_trajectory("g/member-0", "source", 600)
        coordinator.register_owned_running_trajectory(source)
        coordinator.request_owned_trajectory_preemption(source)
        coordinator.mark_owned_trajectory_safe_point(source)
        snapshot_ref = coordinator.commit_trajectory_snapshot(source, snapshot("g", 0, 2, source, input_kwargs(2)[0]), 900)
        # Source allocated this pending member, but died before registration or
        # its first tool/model call. A valid old read must not authorize later
        # writes after coordinator recovery and a new target generation.
        old = coordinator.open_trajectory("g/member-1", "source", 600)
        input0 = coordinator.put_handoff_member_record(encode_handoff_member_record(input_kwargs(2)[0]))
        input1 = coordinator.put_handoff_member_record(encode_handoff_member_record(input_kwargs(2)[1]))
        base0 = {"trajectory_id": "g/member-0", "group_id": "g", "member_index": 0, "group_size": 2}
        base1 = {"trajectory_id": "g/member-1", "group_id": "g", "member_index": 1, "group_size": 2}
        manifest = HandoffManifest("handoff", "source", producer, IDENTITY, "env-v1", "tool-v1", (),
                                   ({**base0, "snapshot_ref": snapshot_ref, "expected_generation": 0, "input_ref": input0},),
                                   (), ({**base1, "record_ref": input1},))
        coordinator.prepare_handoff_manifest(manifest)
        coordinator.commit_handoff_manifest("handoff")
        coordinator.release_handoff_source("handoff")
        coordinator.check_trajectory_owner(old)
        coordinator.close()
        coordinator = DurableTrajectoryCoordinator(str(tmp_path), "run")
        coordinator.begin_handoff_restore("handoff")
        target = coordinator.acquire_handoff_pending("handoff", old.trajectory_id, "target", 600)
        assert target.generation == old.generation + 1
        with pytest.raises(ValueError, match="stale"):
            coordinator.register_owned_running_trajectory(old)
        with pytest.raises(ValueError, match="unknown trajectory"):
            coordinator.get_trajectory_preemption(old.trajectory_id)
        coordinator.register_owned_running_trajectory(target)
        with pytest.raises(ValueError, match="stale"):
            coordinator.request_owned_trajectory_preemption(old)
        assert coordinator.get_trajectory_preemption(target.trajectory_id) == "running"
        assert coordinator.request_owned_trajectory_preemption(target) == "preempt_requested"
        with pytest.raises(ValueError, match="stale"):
            coordinator.mark_owned_trajectory_safe_point(old)
        assert coordinator.get_trajectory_preemption(target.trajectory_id) == "preempt_requested"
        assert coordinator.mark_owned_trajectory_safe_point(target) == "safe_point_reached"
        coordinator.check_trajectory_owner(target)
    finally:
        coordinator.close()
