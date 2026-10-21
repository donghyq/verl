"""Opt-in V1 binding keeps original data and avoids duplicate publication."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.ppo.v1.trajectory_entry import (
    PreparedTrajectoryMember,
    V1TrajectoryEntry,
    assemble_prepared_fields,
    state_settings,
    validate_trajectory_fields,
)
from verl.trainer.ppo.v1.trajectory_state.runtime import PromptGroupResult


def original():
    return {"uid": "prompt", "agent_name": "tool_agent", "raw_prompt": [{"role": "user", "content": "x"}],
            "global_steps": 0, "tools_kwargs": {"search": {"city": "original"}}}


def fields():
    return TensorDict({
        "prompts": torch.tensor([[2, 3]]), "responses": torch.tensor([[5, 7]]),
        "loss_mask": torch.tensor([[1, 0]]), "response_mask": torch.tensor([[1, 0]]),
        "input_ids": torch.tensor([[2, 3, 5, 7]]), "rollout_log_probs": torch.tensor([[-.5, 0.]]),
    }, batch_size=[1])


def worker():
    writes = []

    async def postprocess(output, validate, **kwargs):
        writes.append((output, validate, kwargs))

    async def prepare(output, validate, **kwargs):
        writes.append((output, validate, kwargs))
        return [f"{kwargs['uid']}_{kwargs['session_id']}_0"], fields(), [{"status": "success"}]

    return SimpleNamespace(
        config=OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 8}}}),
        _agent_loop_postprocess=postprocess, _prepare_agent_loop_outputs=prepare, writes=writes,
    )


class Runtime:
    def __init__(self, status="completed"):
        self.status = status
        self.calls = []
        self.policy_identity = {"policy_version": "0"}
        self.environment_version = "env"
        self.tool_version = "tools"

    async def run_source(self, uid, members, callback, *, start_members):
        self.calls.append((uid, members, callback, start_members))
        prepared = []
        if self.status == "completed":
            async def owner():
                return None

            async def commit(snapshot):
                return None

            for i, data in enumerate(members):
                prepared.append(await callback(
                    i, data, SimpleNamespace(trajectory_id=f"{uid}/member-{i}", generation=0), commit, owner
                ))
        return PromptGroupResult(
            status=self.status, outputs=prepared,
            handoff_id="handoff" if self.status == "handed_off" else None,
            member_kwargs=members, output_record_refs={}, metrics={},
        )


@pytest.mark.asyncio
async def test_normal_completion_uses_original_postprocessing_and_preserves_input(monkeypatch):
    import transfer_queue as tq

    queue_writes = []

    async def put(**kwargs):
        queue_writes.append(kwargs)

    class Loop:
        async def run_until_checkpoint(self, sampling_params, **kwargs):
            return kwargs["session_id"]

    monkeypatch.setattr(tq, "async_kv_batch_put", put)
    data = original()
    runtime, target = Runtime(), worker()
    entry = V1TrajectoryEntry(target, runtime)
    entry._new_loop = lambda kwargs: Loop()
    result = await entry.run_prompt(
        data, {"temperature": 1.}, {"validate": False}, trace=True
    )
    assert result.status == "completed"
    assert [row[0] for row in target.writes] == list(range(8))
    assert [row[2]["session_id"] for row in target.writes] == list(range(8))
    assert all(row[2]["tools_kwargs"] == data["tools_kwargs"] for row in target.writes)
    assert all("__trajectory_group_size__" not in row[2] and "sampling_params" not in row[2]
               for row in target.writes)
    assert data == original()
    assert runtime.calls[0][3] is None
    assert [row["keys"] for row in queue_writes] == [[f"prompt_{i}_0"] for i in range(8)]
    assert all(isinstance(item, PreparedTrajectoryMember) for item in result.outputs)


@pytest.mark.asyncio
async def test_checkpoint_result_is_never_postprocessed_as_a_training_sample():
    target, runtime = worker(), Runtime("handed_off")
    result = await V1TrajectoryEntry(target, runtime).run_prompt(
        original(), {"temperature": 1.}, {"validate": False}, start_members=list(range(6))
    )
    assert result.handoff_id == "handoff"
    assert target.writes == []
    assert runtime.calls[0][3] == list(range(6))


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_succeeds", [True, False])
async def test_acceptance_retains_only_the_actual_successfully_committed_source_objects(commit_succeeds):
    snapshot = {"actual_tokens": [1, 2], "actual_member": 2}
    lease = SimpleNamespace(trajectory_id="prompt/member-2", generation=0, lease_id="actual-source-lease")

    class Loop:
        async def run_until_checkpoint(self, sampling_params, *, commit_snapshot, **kwargs):
            await commit_snapshot(snapshot)
            raise RuntimeError("checkpoint yield")

    async def commit(data):
        assert data is snapshot
        if not commit_succeeds:
            raise ValueError("source commit failed")
        return "actual-reference"

    async def check_owner():
        return None

    runtime, target = Runtime(), worker()
    entry = V1TrajectoryEntry(target, runtime)
    assert entry.acceptance_source_records("prompt") is None
    entry.enable_acceptance_source_observation("prompt")
    with pytest.raises(ValueError, match="already enabled"):
        entry.enable_acceptance_source_observation("prompt")
    entry._new_loop = lambda original: Loop()
    kwargs = {**original(), "sampling_params": {}, "__trajectory_group_size__": 8, "session_id": 2}
    with pytest.raises((ValueError, RuntimeError)):
        await entry._run_source_member(2, kwargs, lease, commit, check_owner)
    records = entry.acceptance_source_records("prompt")
    if commit_succeeds:
        assert records[2]["owner_lease"] == lease
        assert records[2]["snapshot"] == snapshot
        assert records[2]["snapshot_ref"] == "actual-reference"
        records[2]["snapshot"]["actual_tokens"].append(999)
        assert entry.acceptance_source_records("prompt")[2]["snapshot"]["actual_tokens"] == [1, 2]
    else:
        assert records == {}
    assert target.writes == []


@pytest.mark.asyncio
async def test_acceptance_snapshot_wrapper_preserves_publication_callbacks():
    seen = []
    lease = SimpleNamespace(trajectory_id="prompt/member-2", generation=0, lease_id="actual-source-lease")

    class Loop:
        async def run_until_checkpoint(self, sampling_params, *, commit_snapshot, **kwargs):
            seen.append(callable(getattr(commit_snapshot, "resolve_published_policy", None)))
            seen.append(callable(getattr(commit_snapshot, "get_snapshot_policy_identity", None)))
            return 2

    async def commit(data):
        return None

    async def resolve(version):
        return {"version": version}

    async def snapshot_identity():
        return {"policy_version": "1"}

    commit.resolve_published_policy = resolve
    commit.get_snapshot_policy_identity = snapshot_identity

    async def check_owner():
        return None

    runtime, target = Runtime(), worker()
    entry = V1TrajectoryEntry(target, runtime)
    entry._new_loop = lambda original: Loop()
    kwargs = {**original(), "sampling_params": {}, "__trajectory_group_size__": 8, "session_id": 2}
    result = await entry._run_source_member(2, kwargs, lease, commit, check_owner)
    assert isinstance(result, PreparedTrajectoryMember)
    assert seen == [True, True]


@pytest.mark.asyncio
async def test_acceptance_boundary_waits_for_release_and_reports_ready_status():
    reached = asyncio.Event()
    released = asyncio.Event()
    lease = SimpleNamespace(trajectory_id="prompt/member-2", generation=0, lease_id="actual-source-lease")

    class Loop:
        async def run_until_checkpoint(self, sampling_params, **kwargs):
            callback = kwargs["_trajectory_acceptance_tool_boundary"]
            await callback(0, 1)
            reached.set()
            released.set()
            return 2

    async def commit(data):
        return None

    async def check_owner():
        return None

    runtime, target = Runtime(), worker()
    entry = V1TrajectoryEntry(target, runtime)
    entry.enable_acceptance_source_observation("prompt", checkpoint_members=[2, 3, 4, 5])
    entry._new_loop = lambda original: Loop()
    kwargs = {**original(), "sampling_params": {}, "__trajectory_group_size__": 8, "session_id": 2}
    task = asyncio.create_task(entry._run_source_member(2, kwargs, lease, commit, check_owner))
    for _ in range(100):
        status = entry.acceptance_boundary_status("prompt")
        if status.get("acceptance_boundary_ready_members") == [2]:
            break
        await asyncio.sleep(0.001)
    else:
        raise AssertionError("acceptance boundary did not report ready member")
    assert status == {
        "acceptance_boundary_members": [2, 3, 4, 5],
        "acceptance_boundary_ready_members": [2],
        "acceptance_boundary_released": False,
    }
    entry.release_acceptance_boundaries("prompt")
    result = await task
    assert isinstance(result, PreparedTrajectoryMember)
    assert reached.is_set() and released.is_set()
    assert entry.acceptance_boundary_status("prompt")["acceptance_boundary_released"] is True


@pytest.mark.asyncio
async def test_acceptance_boundary_is_not_injected_by_default():
    seen = []
    lease = SimpleNamespace(trajectory_id="prompt/member-2", generation=0, lease_id="actual-source-lease")

    class Loop:
        async def run_until_checkpoint(self, sampling_params, **kwargs):
            seen.append("_trajectory_acceptance_tool_boundary" in kwargs)
            return 2

    async def commit(data):
        return None

    async def check_owner():
        return None

    runtime, target = Runtime(), worker()
    entry = V1TrajectoryEntry(target, runtime)
    entry._new_loop = lambda original: Loop()
    kwargs = {**original(), "sampling_params": {}, "__trajectory_group_size__": 8, "session_id": 2}
    result = await entry._run_source_member(2, kwargs, lease, commit, check_owner)
    assert isinstance(result, PreparedTrajectoryMember)
    assert seen == [False]
    assert entry.acceptance_boundary_status("prompt") == {}


@pytest.mark.parametrize("members", [None, [], [2, 2], [8], [-1], ["2"]])
def test_acceptance_boundary_rejects_invalid_member_lists(members):
    entry = V1TrajectoryEntry(worker(), Runtime())
    if members is None:
        entry.enable_acceptance_source_observation("prompt")
        assert entry.acceptance_boundary_status("prompt") == {}
        return
    with pytest.raises(ValueError, match="checkpoint members"):
        entry.enable_acceptance_source_observation("prompt", checkpoint_members=members)


@pytest.mark.asyncio
async def test_resume_passes_history_once_without_duplicate_explicit_kwargs():
    calls = []

    class Loop:
        async def resume_from_snapshot(self, snapshot, *, check_owner, execution_generation,
                                       environment_version, tool_version, **kwargs):
            calls.append((snapshot, execution_generation, environment_version, tool_version, kwargs))
            assert not {"_trajectory_checkpoint", "sampling_params", "trajectory"} & set(kwargs)
            await check_owner()
            return "output"

    runtime = SimpleNamespace(policy_identity={"policy_version": "0"}, environment_version="env",
                              tool_version="tools")
    entry = V1TrajectoryEntry(worker(), runtime)
    entry._new_loop = lambda kwargs: Loop()
    kwargs = {**original(), "sampling_params": {"temperature": 1.}, "trajectory": {"validate": False},
              "__trajectory_group_size__": 8, "session_id": 2}

    async def check_owner():
        calls.append("owner_checked")

    snapshot = SimpleNamespace(policy_identity={
        "policy_version": "1",
        "model_weights_fingerprint": "weights-1",
        "tokenizer_template_fingerprint": "tok-1",
        "kv_cache_schema_fingerprint": "kv-1",
    })
    output = await entry._resume_member(
        2, snapshot, kwargs, SimpleNamespace(trajectory_id="prompt/member-2", generation=1), check_owner
    )
    assert isinstance(output, PreparedTrajectoryMember) and output.output == "output"
    assert calls[0][0] is snapshot and calls[0][1:4] == (1, "env", "tools")
    assert calls[0][4]["raw_prompt"] == kwargs["raw_prompt"]
    assert calls[0][4]["rollout_sample_index"] == 2
    assert calls[-1] == "owner_checked"


@pytest.mark.asyncio
async def test_finished_handoff_never_recreates_consumed_queue_keys(monkeypatch):
    async def reconcile(**kwargs):
        assert kwargs["handoff_id"] == "handoff"
        return {"state": "FINISHED", "already_finished": True, "write_attempted": False,
                "prompt_ready_verified": True, "readback_verified": True}

    monkeypatch.setattr(
        "verl.trainer.ppo.v1.trajectory_publication.reconcile_trajectory_group_publication", reconcile
    )
    class Client:
        async def summary(self):
            return {"handoffs": {"handoff": "finished"}}

        async def get_publication(self, hid):
            return {"state": "PUBLISHED", "receipt": {"actual": "already verified"}}

    class Runtime:
        client = Client()

        async def run_target(self, *args):
            raise AssertionError("finished snapshots can already have been collected")

    result, receipt = await V1TrajectoryEntry(worker(), Runtime()).resume_handoff("handoff")
    assert result is None
    assert receipt["already_finished"] and not receipt["write_attempted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("target_version", ["0", "1"])
async def test_production_resume_selects_token_recompute_only_after_policy_changes(monkeypatch, target_version):
    from verl.trainer.ppo.v1.trajectory_state import (
        CanonicalPolicyKVIdentity,
        EnginePolicyObservation,
        PublishedPolicyRecord,
    )

    source_identity = {"policy_version": "0", "model_weights_fingerprint": "weights-0",
                       "tokenizer_template_fingerprint": "template", "kv_cache_schema_fingerprint": "kv"}
    active_identity = {**source_identity, "policy_version": target_version,
                       "model_weights_fingerprint": "weights-" + target_version}
    calls = []
    members = [{**original(), "session_id": i} for i in range(8)]
    outputs = [PreparedTrajectoryMember(i, [f"prompt_{i}_0"], fields(), [{"status": "success"}]) for i in range(8)]

    class Client:
        async def summary(self):
            return {"handoffs": {"handoff": "committed"}}

        async def get_publication(self, hid):
            return None

        async def get_handoff_manifest(self, hid):
            return SimpleNamespace(policy_identity=source_identity)

        async def get_active_published_policy(self):
            return PublishedPolicyRecord(
                "run", CanonicalPolicyKVIdentity(**active_identity), "fixture-config", "fixture-checkpoint",
                (EnginePolicyObservation("http://fixture", target_version),), 0.,
            ).encode()

    class Runtime:
        client = Client()
        published_policy_client = client

        async def run_target(self, hid, resume, pending, **kwargs):
            calls.append(kwargs)
            return PromptGroupResult("completed", outputs, hid, members, {}, {})

    async def publish(**kwargs):
        assert kwargs["fields"].batch_size[0] == 8
        return {"state": "FINISHED", "prompt_ready_verified": True, "readback_verified": True}

    monkeypatch.setattr("verl.trainer.ppo.v1.trajectory_publication.publish_trajectory_group", publish)
    result, receipt = await V1TrajectoryEntry(worker(), Runtime()).resume_handoff("handoff")
    assert calls == ([{}] if target_version == "0" else [{"token_only": True}])
    assert result.member_kwargs == members and receipt["state"] == "FINISHED"


@pytest.mark.parametrize("error", ["mask", "input", "logprob"])
def test_real_training_field_alignment_rejects_corruption(error):
    data = fields()
    if error == "mask":
        data["loss_mask"][0, 1] = 1
    elif error == "input":
        data["input_ids"][0, 3] = 999
    else:
        data["rollout_log_probs"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        validate_trajectory_fields(data)


def test_valid_zero_tool_logprob_is_kept_for_original_loss_mask():
    validate_trajectory_fields(fields())


def test_eight_different_length_outcomes_keep_exact_tokens_and_non_tensor_fields():
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    original_rows, prepared = [], []
    for i in range(8):
        prompt = torch.arange(2 + i, dtype=torch.int64)
        response = torch.arange(3 + i, dtype=torch.int64) + 10
        row = {"prompts": prompt, "responses": response, "input_ids": torch.cat((prompt, response)),
               "loss_mask": torch.ones_like(response), "response_mask": torch.ones_like(response),
               "rollout_log_probs": torch.full((len(response),), -.5),
               "uid": "prompt", "session_id": i, "extra_fields": {"member": i, "list": [i]}}
        original_rows.append(row)
        prepared.append(list_of_dict_to_tensordict([row]))
    combined = assemble_prepared_fields(prepared)
    validate_trajectory_fields(combined)
    for row, expected in zip(combined.unbind(0), original_rows, strict=True):
        assert torch.equal(row["prompts"], expected["prompts"])
        assert torch.equal(row["responses"], expected["responses"])
        assert row.get_non_tensor("extra_fields") == expected["extra_fields"]
        assert row.get_non_tensor("session_id") == expected["session_id"]


def test_disabled_state_entry_does_not_require_identity_or_data():
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {}}})
    assert state_settings(config) is None


def test_enabled_state_entry_rejects_missing_real_identity():
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"trajectory_state": {"enabled": True}}}})
    with pytest.raises(ValueError, match="root_dir"):
        state_settings(config)
