"""CPU protocol checks for the real Rollouter park/resume/publication entry.

Generation is a deterministic test double. These tests verify tensor and
membership preservation; they do not stand in for the cluster Tool Agent
or optimizer Gate.
"""
import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import ray.cloudpickle as cloudpickle
import torch
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.experimental.fully_async_policy.fully_async_main import (
    validate_group_workload_scheduling_config,
    validate_preemptible_handoff_config,
)
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.handoff_executor import decode_handoff_member_record
from verl.experimental.fully_async_policy.rollout_artifact import CanonicalPolicyKVIdentity
from verl.protocol import DataProto
from tests.experimental.fully_async_policy.test_handoff_executor import FakeHandoffClient, make_snapshot
from tests.experimental.fully_async_policy.test_handoff_wiring_e2e import FakeRayClient

IDENTITY = CanonicalPolicyKVIdentity("0", "weights", "template", "kv")
Rollouter = FullyAsyncRollouter.__ray_metadata__.modified_class


def objects(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


class CompleteClient(FakeHandoffClient):
    open_trajectory = FakeRayClient.open_trajectory
    register_running_trajectory = FakeRayClient.register_running_trajectory
    get_trajectory_preemption = FakeRayClient.get_trajectory_preemption
    mark_trajectory_safe_point = FakeRayClient.mark_trajectory_safe_point
    commit_trajectory_snapshot = FakeRayClient.commit_trajectory_snapshot
    mark_trajectory_snapshot_committed = FakeRayClient.mark_trajectory_snapshot_committed
    check_trajectory_owner = FakeRayClient.check_trajectory_owner
    release_trajectory_owner = FakeRayClient.release_trajectory_owner
    request_trajectory_preemption = FakeRayClient.request_trajectory_preemption
    put_handoff_member_record = FakeRayClient.put_handoff_member_record
    prepare_handoff_manifest = FakeRayClient.prepare_handoff_manifest
    commit_handoff_manifest = FakeRayClient.commit_handoff_manifest
    record_handoff_source_release = FakeRayClient.record_handoff_source_release

    async def find_handoff_artifact(self, handoff_id, artifact_id):
        return self.restore.find_artifact(handoff_id, artifact_id)

    async def publish(self, manifest, payload):
        self.store.publish(manifest, payload)


class Queue:
    def __init__(self):
        self.notifications = []

    async def put_sample(self, sample):
        self.notifications.append(sample)
        return True


def make_rollouter(client):
    actor = object.__new__(Rollouter)
    actor.config = OmegaConf.create({"async_training": {
        "preemptible_handoff": {"enabled": True, "lease_seconds": 600},
        "artifact_data_plane": {"environment_version": "env-1", "tool_version": "tools-1"},
    }})
    actor.artifact_store_client = client
    actor.message_queue_client = Queue()
    actor._producer_generation = client.generation
    actor._handoff_quiesced = False
    actor._parked_handoff_samples = {}
    actor.paused = False
    actor.lock = asyncio.Lock()
    actor._resume_event = asyncio.Event()
    actor.total_generated_samples = actor.dropped_stale_samples = actor.processed_sample_count = 0
    actor._policy_kv_identity = lambda version: IDENTITY

    async def serving_version():
        return "0"  # fixture of an applied serving receipt, not a sample count

    actor._get_active_serving_policy_version = serving_version

    async def statistics():
        return {}

    actor.get_statistics = statistics
    return actor


def request_batch():
    return DataProto.from_dict(
        tensors={"attention_mask": torch.ones(8, 2, dtype=torch.int64)},
        non_tensors={"raw_prompt": objects([[{"role": "user", "content": f"member-{i}"}] for i in range(8)]),
                     "extra_info": objects([{"original_member": i} for i in range(8)])},
        meta_info={"validate": False},
    )


def output_row(index, extra=None):
    return DataProto.from_dict(
        tensors={"responses": torch.tensor([[100 + index, 200 + index, 0]]),
                 "response_mask": torch.tensor([[1, 1, 0]]),
                 "rollout_log_probs": torch.tensor([[-0.1 - index, -0.2 - index, 0.0]])},
        non_tensors={"rollout_sample_index": np.array([index], dtype=np.int64),
                     "max_global_steps": np.array([0], dtype=np.int64),
                     "extra_info": objects([{"original_member": index}]),
                     **(extra or {})},
        meta_info={"metrics": [{"generate_sequences": float(index + 1), "tool_calls": index / 10}]},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted", [[0, 1, 2, 3, 4, 5], [0, 1, 4, 5, 6, 7]])
async def test_actual_rollouter_preserves_completed_tensors_and_first_starts_pending(admitted):
    client = CompleteClient()
    source = make_rollouter(client)
    checkpointed = admitted[2:]
    pending = sorted(set(range(8)) - set(admitted))
    target_runs = []

    async def source_generate(batch):
        assert len(batch) == 6
        assert [int(i) for i in batch.non_tensor_batch["__trajectory_sample_index__"]] == admitted
        assert not any(f"group-1#{i}" in client.snapshots.owners for i in pending)
        await source.request_preemptible_handoff([f"group-1#{i}" for i in checkpointed])
        rows = []
        for row_index, index in enumerate(admitted):
            extras = {"__preempted__": objects([index in checkpointed]),
                      "__preemption_snapshot_ref__": objects([None]),
                      "__preemption_trajectory_id__": objects([None]),
                      "__preemption_expected_generation__": objects([None])}
            if index in checkpointed:
                lease = client.snapshots.owners[f"group-1#{index}"]
                snapshot = make_snapshot(f"group-1#{index}", "group-1", index, 8, lease.generation)
                ref = await batch.non_tensor_batch["__trajectory_checkpoint__"][row_index](snapshot)
                extras["__preemption_snapshot_ref__"] = objects([ref])
                extras["__preemption_trajectory_id__"] = objects([snapshot.trajectory_id])
                extras["__preemption_expected_generation__"] = objects([lease.generation])
            rows.append(output_row(index, extras))
        output = DataProto.concat(rows)
        output.meta_info["metrics"] = [row.meta_info["metrics"][0] for row in rows]
        return output

    source.async_rollout_manager = SimpleNamespace(generate_sequences_single=source_generate)
    await source._process_single_sample_streaming(
        RolloutSample(request_batch(), "group-1", 7, {}), initial_member_indices=admitted)
    manifest = await client.get_handoff_manifest("handoff-group-1")
    tables = manifest.member_tables()["group-1"]
    assert tables["completed"] == [0, 1]
    assert tables["checkpointed"] == sorted(checkpointed)
    assert tables["pending"] == pending
    assert client.handoffs.get(manifest.handoff_id).state.value == "committed"
    assert not source.message_queue_client.notifications
    for member in manifest.completed_members:
        saved = decode_handoff_member_record(await client.get_handoff_member_record(member["record_ref"]))
        assert torch.equal(saved.batch["responses"], output_row(member["member_index"]).batch["responses"])
        assert torch.equal(saved.batch["rollout_log_probs"], output_row(member["member_index"]).batch["rollout_log_probs"])
        assert saved.meta_info["metrics"] == output_row(member["member_index"]).meta_info["metrics"]

    target = make_rollouter(client)

    async def target_generate(batch):
        index = int(batch.non_tensor_batch["__trajectory_sample_index__"][0])
        target_runs.append(index)
        assert batch.non_tensor_batch["extra_info"][0]["original_member"] == index
        assert batch.non_tensor_batch["raw_prompt"][0][0]["content"] == f"member-{index}"
        assert batch.non_tensor_batch["__trajectory_execution_generation__"][0] == (1 if index in checkpointed else 0)
        assert ("__trajectory_resume__" in batch.non_tensor_batch) == (index in checkpointed)
        await batch.non_tensor_batch["__trajectory_check_owner__"][0]()
        return output_row(index)

    target.async_rollout_manager = SimpleNamespace(generate_sequences_single=target_generate)
    receipt = await target.resume_preempted_group("handoff-group-1", owner_id="target")
    assert target_runs == sorted(checkpointed) + pending
    assert receipt["artifact_id"] == "group-1"
    assert receipt["outcome"]["tool_replay_count"] is None
    assert receipt["outcome"]["recomputed_prefill_tokens"] is None
    assert client.handoffs.get(manifest.handoff_id).state.value == "finished"
    assert len(target.message_queue_client.notifications) == 1
    assert not client.snapshots.latest
    lease = client.store.claim_complete_group(accepted_policy_versions={"0"}, lease_timeout_seconds=30)
    sample = cloudpickle.loads(lease.payloads[0])
    assert sample.epoch == 7 and len(sample.full_batch) == 8
    assert sample.full_batch.batch["responses"][:, 0].tolist() == list(range(100, 108))
    assert sample.full_batch.non_tensor_batch["rollout_sample_index"].tolist() == list(range(8))
    assert sample.full_batch.meta_info["metrics"] == [output_row(i).meta_info["metrics"][0] for i in range(8)]
    from verl.experimental.fully_async_policy.detach_utils import addition_process
    assembled = addition_process(sample.full_batch)
    assert assembled.non_tensor_batch["processing_times"] == list(range(1, 9))
    assert not any(key.startswith("__preemption_") or key == "__preempted__" for key in sample.full_batch.non_tensor_batch)
    client.store.acknowledge(lease)
    assert client.store.claim_complete_group(accepted_policy_versions={"0"}, lease_timeout_seconds=30) is None


def test_opt_in_handoff_config_requires_real_artifact_semantics():
    config = OmegaConf.create({"async_training": {"preemptible_handoff": {"enabled": True}}})
    with pytest.raises(ValueError, match="training-semantic"):
        validate_preemptible_handoff_config(config)
    config.async_training.preemptible_handoff.enabled = False
    validate_preemptible_handoff_config(config)


def test_opt_in_scheduling_requires_calibration_and_defaults_to_off():
    config = OmegaConf.create({"async_training": {}})
    validate_group_workload_scheduling_config(config)
    config.async_training.group_workload_scheduling = {"enabled": True}
    with pytest.raises(ValueError, match="independent calibration"):
        validate_group_workload_scheduling_config(config)
    config.async_training.group_workload_scheduling.calibration_path = "independent.json"
    validate_group_workload_scheduling_config(config)


class RemoteCall:
    def __init__(self, callback):
        self.callback = callback

    async def remote(self, *args, **kwargs):
        return self.callback(*args, **kwargs)


@pytest.mark.asyncio
async def test_serving_version_requires_actual_consistent_receipts():
    actor = object.__new__(Rollouter)
    servers = {name: SimpleNamespace(get_global_steps=RemoteCall(lambda version=version: version))
               for name, version in [("a", 9), ("b", 9)]}
    actor.llm_server_manager = SimpleNamespace(global_load_balancer=SimpleNamespace(
        get_server_handles=RemoteCall(lambda: servers)))
    assert await actor._get_active_serving_policy_version() == "9"
    servers["b"].get_global_steps = RemoteCall(lambda: 10)
    with pytest.raises(ValueError, match="consistent applied"):
        await actor._get_active_serving_policy_version()


@pytest.mark.asyncio
async def test_group_registration_uses_all_input_members_before_rpc_and_off_is_unchanged():
    actor = make_rollouter(CompleteClient())
    registered = []
    actor.llm_server_manager = SimpleNamespace(global_load_balancer=SimpleNamespace(
        register_group=RemoteCall(lambda contexts: registered.append(contexts) or True)))
    batch = request_batch()
    sample = RolloutSample(batch, "group-placement", 0, {})
    assert not await actor._register_group_placement(sample)
    assert not registered and "__group_route_context__" not in batch.non_tensor_batch
    actor.config.async_training.group_workload_scheduling = {"enabled": True}
    actor.config.actor_rollout_ref = {"rollout": {"response_length": 128}}
    batch.non_tensor_batch["extra_info"][0]["workload_key"] = "short"
    assert await actor._register_group_placement(sample)
    assert [context.member_index for context in registered[0]] == list(range(8))
    assert registered[0][0].workload_key == "short" and registered[0][1].workload_key == "unknown"
    assert all(context.prompt_tokens == 2 and context.max_new_tokens == 128 for context in registered[0])
    assert len(batch.non_tensor_batch["__group_route_context__"]) == 8
