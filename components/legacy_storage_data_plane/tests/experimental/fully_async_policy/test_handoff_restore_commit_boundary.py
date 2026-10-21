"""Regressions for the target interruption and actual publication boundary."""
import json
from dataclasses import asdict
from dataclasses import replace

import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.experimental.fully_async_policy.handoff_executor import publish_restored_group, restore_handoff
from verl.experimental.fully_async_policy.preemptible_handoff import (
    HandoffManifest,
    HandoffState,
    decode_handoff_member_record,
    encode_handoff_member_record,
)
from tests.experimental.fully_async_policy.test_handoff_executor import (
    FakeHandoffClient,
    build_n8_handoff,
    fake_resume,
)


def test_member_record_preserves_tensors_and_training_provenance():
    row = DataProto.from_dict(
        tensors={"responses": torch.tensor([[1, 2, 3]]),
                 "response_mask": torch.tensor([[1, 0, 1]]),
                 "rollout_log_probs": torch.tensor([[-0.1, 0.0, -0.2]])},
        non_tensors={"extra_info": np.array([{"trajectory_segments": [{"execution_generation": 0}],
                                               "reward_model": {"ground_truth": "real input"}}], dtype=object)})
    decoded = decode_handoff_member_record(encode_handoff_member_record(row))
    for key in row.batch.keys():
        assert torch.equal(row.batch[key], decoded.batch[key])
    assert decoded.non_tensor_batch["extra_info"][0] == row.non_tensor_batch["extra_info"][0]
    broken = json.loads(encode_handoff_member_record(row))
    broken["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        decode_handoff_member_record(json.dumps(broken).encode())


@pytest.mark.asyncio
async def test_source_generation_is_rechecked_at_manifest_commit():
    client = FakeHandoffClient()
    await build_n8_handoff(client)
    record = client.handoffs.get("handoff-n8")
    record.state = HandoffState.PREPARING
    client.store.register_producer("fully_async_rollouter", "new-incarnation")
    with pytest.raises(ValueError, match="generation changed"):
        client.handoffs.commit("handoff-n8")
    assert record.state is HandoffState.PREPARING


@pytest.mark.asyncio
async def test_manifest_accepts_only_known_complete_input_references():
    client = FakeHandoffClient()
    await build_n8_handoff(client)
    original = client.handoffs.get("handoff-n8").manifest
    draft = HandoffManifest(**{**asdict(original),
        "checkpointed_trajectories": tuple({**m, "input_ref": "handoff-member/source-input"}
                                            for m in original.checkpointed_trajectories),
        "pending_samples": tuple({**m, "record_ref": "handoff-member/pending-input"}
                                  for m in original.pending_samples)})
    assert HandoffManifest.decode(draft.encode()) == draft
    bad = HandoffManifest(**{**asdict(draft),
        "pending_samples": ({**draft.pending_samples[0], "record_ref": ""},) + draft.pending_samples[1:]})
    with pytest.raises(ValueError, match="record_ref"):
        bad.validate()
    unknown = HandoffManifest(**{**asdict(draft),
        "checkpointed_trajectories": ({**draft.checkpointed_trajectories[0], "unknown": 1},)
                                      + draft.checkpointed_trajectories[1:]})
    with pytest.raises(ValueError, match="unexpected fields"):
        unknown.validate()


@pytest.mark.asyncio
async def test_target_death_mid_restore_preserves_results_and_reuses_current_generation():
    client = FakeHandoffClient()
    await build_n8_handoff(client)
    calls = []

    async def die_on_second_member(member, snapshot):
        calls.append(member["member_index"])
        if member["member_index"] == 3:
            raise RuntimeError("target died during member 3")
        return await fake_resume(member, snapshot)

    with pytest.raises(RuntimeError, match="target died"):
        await restore_handoff(client, "handoff-n8", owner_id="target-a", lease_seconds=1,
                              resume_trajectory=die_on_second_member)
    assert calls == [2, 3]
    assert set(client.handoffs.get("handoff-n8").restored_member_refs) == {2}
    assert len(client.snapshots.latest) == 4  # nothing deleted before publication
    generation_after_first_takeover = client.snapshots.owners["group-1#3"].generation
    with pytest.raises(ValueError, match="expired matching owner"):
        await restore_handoff(client, "handoff-n8", owner_id="target-b", resume_trajectory=fake_resume)
    clock = client.snapshots.clock
    client.snapshots.clock = lambda: clock() + 2
    resumed_calls = []

    async def resume_second_target(member, snapshot):
        resumed_calls.append(member["member_index"])
        assert member["owner_lease"].generation > snapshot.execution_generation
        return await fake_resume(member, snapshot)

    outcome, records = await restore_handoff(
        client, "handoff-n8", owner_id="target-b", resume_trajectory=resume_second_target)
    assert resumed_calls == [3, 4, 5]  # completed member 2 never runs twice
    assert client.snapshots.owners["group-1#3"].generation == generation_after_first_takeover + 1
    assert sorted(records) == list(range(6))
    pending = {i: {"member_index": i} for i in (6, 7)}
    await publish_restored_group(client, outcome, records, pending, publish_group=client.publish_fixture_group)
    assert client.handoffs.get("handoff-n8").state is HandoffState.FINISHED
    assert not client.snapshots.latest
    assert client.store.claim_complete_group(accepted_policy_versions={"0"}, lease_timeout_seconds=30)


@pytest.mark.asyncio
async def test_failed_publication_keeps_resume_results_and_checkpoints():
    client = FakeHandoffClient()
    await build_n8_handoff(client)
    outcome, records = await restore_handoff(client, "handoff-n8", resume_trajectory=fake_resume)
    pending = {i: {"member_index": i} for i in (6, 7)}

    async def fail_publish(merged):
        raise RuntimeError("payload PUT failed")

    with pytest.raises(RuntimeError, match="PUT failed"):
        await publish_restored_group(client, outcome, records, pending, publish_group=fail_publish)
    assert client.handoffs.get("handoff-n8").state is HandoffState.RESTORING
    assert len(client.snapshots.latest) == 4

    async def must_not_resume(member, snapshot):
        raise AssertionError("saved result should prevent duplicate execution")

    retry_outcome, retry_records = await restore_handoff(
        client, "handoff-n8", resume_trajectory=must_not_resume)
    await publish_restored_group(client, retry_outcome, retry_records, pending,
                                 publish_group=client.publish_fixture_group)
    lease = client.store.claim_complete_group(accepted_policy_versions={"0"}, lease_timeout_seconds=30)
    assert lease is not None
    client.store.acknowledge(lease)
    assert client.store.claim_complete_group(accepted_policy_versions={"0"}, lease_timeout_seconds=30) is None


@pytest.mark.asyncio
async def test_fabricated_publication_id_does_not_finish_or_collect_snapshots():
    client = FakeHandoffClient()
    await build_n8_handoff(client)
    outcome, records = await restore_handoff(client, "handoff-n8", resume_trajectory=fake_resume)

    async def pretend_publish(merged):
        return "missing-artifact"

    with pytest.raises(ValueError, match="actual complete-group"):
        await publish_restored_group(client, outcome, records, {6: {}, 7: {}}, publish_group=pretend_publish)
    assert client.handoffs.get("handoff-n8").state is HandoffState.RESTORING
    assert len(client.snapshots.latest) == 4


@pytest.mark.asyncio
async def test_expired_result_owner_cannot_rewrite_restored_payload():
    client = FakeHandoffClient()
    await build_n8_handoff(client)
    await restore_handoff(client, "handoff-n8", lease_seconds=1, resume_trajectory=fake_resume)
    lease = client.snapshots.owners["group-1#2"]
    clock = client.snapshots.clock
    client.snapshots.clock = lambda: clock() + 2
    with pytest.raises(ValueError, match="stale or expired"):
        await client.record_handoff_member_result(
            "handoff-n8", 2, lease, encode_handoff_member_record({"response_ids": [999]}), {})


@pytest.mark.asyncio
async def test_restore_rejects_policy_identity_change_before_execution():
    client = FakeHandoffClient()
    await build_n8_handoff(client)
    original = client.handoffs.get("handoff-n8").manifest
    changed = HandoffManifest(**{**asdict(original),
                                 "policy_identity": {**original.policy_identity, "policy_version": "999"}})
    client.handoffs.get("handoff-n8").manifest = changed
    with pytest.raises(ValueError, match="policy identity"):
        await restore_handoff(client, "handoff-n8", resume_trajectory=fake_resume)


@pytest.mark.asyncio
async def test_finished_pending_member_is_returned_without_running_again():
    client = FakeHandoffClient()
    await build_n8_handoff(client)
    await restore_handoff(client, "handoff-n8", resume_trajectory=fake_resume)
    lease = await client.acquire_handoff_pending("handoff-n8", "group-1#6", "target-a", 600)
    result = {"member_index": 6, "response_ids": [41, 42], "tool_calls_executed": 1}
    await client.record_handoff_member_result("handoff-n8", 6, lease,
                                               encode_handoff_member_record(result), {"tool_replay_count": 0})
    client.snapshots.finish(lease)

    async def must_not_resume(member, snapshot):
        raise AssertionError("saved member must not execute again")

    outcome, records = await restore_handoff(client, "handoff-n8", resume_trajectory=must_not_resume)
    assert outcome.pending_members == [7]
    assert outcome.finished_pending_members == [6]
    assert records[6] == result
    with pytest.raises(ValueError, match="already completed"):
        await client.acquire_handoff_pending("handoff-n8", "group-1#6", "target-b", 600)


@pytest.mark.asyncio
async def test_uncommitted_pending_retry_is_explicit_restart_with_fencing():
    client = FakeHandoffClient()
    await build_n8_handoff(client)
    await restore_handoff(client, "handoff-n8", resume_trajectory=fake_resume)
    old = await client.acquire_handoff_pending("handoff-n8", "group-1#7", "target-a", 1)
    with pytest.raises(ValueError, match="live owner"):
        await client.acquire_handoff_pending("handoff-n8", "group-1#7", "target-b", 600)
    clock = client.snapshots.clock
    client.snapshots.clock = lambda: clock() + 2
    new = await client.acquire_handoff_pending("handoff-n8", "group-1#7", "target-b", 600)
    assert new.generation == old.generation + 1
    assert "group-1#7" not in client.snapshots.latest  # no fabricated recovery point
    with pytest.raises(ValueError, match="stale or expired"):
        client.snapshots._check(old)


@pytest.mark.asyncio
async def test_publication_lookup_reuses_actual_submission_and_rejects_wrong_identity():
    client = FakeHandoffClient()
    await build_n8_handoff(client)
    _, records = await restore_handoff(client, "handoff-n8", resume_trajectory=fake_resume)
    assert client.restore.find_artifact("handoff-n8", "artifact-id-1") is None
    await client.publish_fixture_group({**records, 6: {}, 7: {}})
    actual = client.store._artifacts["artifact-id-1"]
    immutable_payload = actual.payload
    assert client.restore.find_artifact("handoff-n8", "artifact-id-1") == {
        "artifact_id": "artifact-id-1", "status": "ready"}
    assert actual.payload is immutable_payload
    original = actual.manifest
    actual.manifest = replace(original, identity=replace(original.identity, policy_version="wrong"))
    with pytest.raises(ValueError, match="does not match"):
        client.restore.find_artifact("handoff-n8", "artifact-id-1")


@pytest.mark.asyncio
async def test_missing_runtime_observations_remain_unknown():
    client = FakeHandoffClient()
    await build_n8_handoff(client)

    async def unobserved_resume(member, snapshot):
        record, _ = await fake_resume(member, snapshot)
        return record, {"tool_replay_count": None}

    outcome, _ = await restore_handoff(client, "handoff-n8", resume_trajectory=unobserved_resume)
    assert outcome.resume_input_tokens == 24
    assert outcome.recomputed_prefill_tokens is None
    assert outcome.tool_replay_count is None
