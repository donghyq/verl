"""CPU contracts for saved Artifact recovery; fixtures are not GPU evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import ray.cloudpickle as cloudpickle
import torch

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.experimental.fully_async_policy.rollout_artifact import (
    CanonicalPolicyKVIdentity,
    MooncakeRolloutArtifactStore,
    RolloutArtifactManifest,
    rollout_training_semantics_from_batch,
    validate_rollout_training_payload,
)


@pytest.fixture
def helper(monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "project-b-final-validation-20260928/terminal-20261007"
    monkeypatch.syspath_prepend(str(directory))
    from gate1_saved_trainer_input import publish_saved_training_artifact, validate_saved_training_artifact
    return SimpleNamespace(validate=validate_saved_training_artifact, publish=publish_saved_training_artifact)


class PayloadStore:
    def __init__(self):
        self.objects, self.puts, self.gets = {}, [], []
        self.pending_removal_count = 0

    def put(self, key, value):
        self.puts.append((key, value))
        self.objects[key] = value

    def exists(self, key):
        return key in self.objects

    def get(self, key):
        self.gets.append(key)
        return self.objects[key]

    def remove(self, key):
        self.objects.pop(key, None)

    def retry_pending_removals(self):
        pass


class PublicClient:
    """Async transport fixture; every operation is an original public store API."""
    def __init__(self, store):
        self.store = store

    async def export_metadata_snapshot(self):
        return self.store.export_metadata_snapshot()

    async def restore_metadata_snapshot(self, snapshot):
        self.store.restore_metadata_snapshot(snapshot)

    async def get_current_producer_generation(self, producer_id):
        return self.store.get_current_producer_generation(producer_id)

    async def publish(self, manifest, payload):
        self.store.publish(manifest, payload)


def saved_fixture(*, corrupt_final_reward=False, placeholder=False):
    metadata = {
        "min_global_steps": np.ones(8, dtype=np.int64),
        "max_global_steps": np.ones(8, dtype=np.int64),
        "producer_generation": np.ones(8, dtype=np.int64),
        "rollout_group_id": np.array(["gate1-handoff-n8"] * 8, dtype=object),
        "rollout_group_size": np.full(8, 8, dtype=np.int64),
        "rollout_sample_index": np.arange(8),
        "rollout_group_role": np.array(["sample"] * 8, dtype=object),
        "uid": np.array(["uid_gate1-handoff-n8"] * 8, dtype=object),
    }
    segments = np.empty(8, dtype=object)
    for index in range(8):
        segments[index] = [{"start": 0, "end": 2, "policy_version": "1",
                            "sampling_config_fingerprint": "a" * 64,
                            "turn_index": 1, "execution_generation": 1}]
    metadata["trajectory_segments"] = segments
    if placeholder:
        metadata["__preempted__"] = np.full(8, False)
    rewards = torch.zeros(8, 2)
    if corrupt_final_reward:
        rewards[2, 1] = float("nan")
    batch = DataProto.from_dict(
        tensors={"responses": torch.arange(16).reshape(8, 2) + 100,
                 "response_mask": torch.ones(8, 2, dtype=torch.int64),
                 "rollout_log_probs": torch.full((8, 2), -0.25), "rm_scores": rewards},
        non_tensors=metadata, meta_info={"metrics": [{} for _ in range(8)]},
    )
    payload = cloudpickle.dumps(RolloutSample(batch, "gate1-handoff-n8", 0, {}))
    manifest = RolloutArtifactManifest(
        artifact_id="gate1-handoff-n8", producer_id="source",
        identity=CanonicalPolicyKVIdentity("1", "weights-real-fixture", "tokenizer-fixture", "schema-fixture"),
        rollout_group_id="gate1-handoff-n8", rollout_group_size=1,
        rollout_sample_index=0, trajectory_count=8,
        training_semantics=rollout_training_semantics_from_batch(
            batch, policy_version="1", environment_version="env-fixture", tool_version="tool-fixture",
            producer_generation=1,
        ),
    )
    original = MooncakeRolloutArtifactStore(namespace="fixture/original", payload_store=PayloadStore())
    assert original.register_producer("source", "original-A") == 0
    assert original.register_producer("source", "original-B") == 1
    original.publish(manifest, payload)
    return payload, original.export_metadata_snapshot()


def validate(helper, payload, metadata):
    return helper.validate(payload, metadata, environment_version="env-fixture", tool_version="tool-fixture",
                           accepted_policy_versions={"0", "1"})


@pytest.mark.asyncio
async def test_original_manifest_and_generation_restored_then_native_put_claim_ack(helper):
    payload, metadata = saved_fixture()
    source_before = copy.deepcopy(metadata)
    saved = validate(helper, payload, metadata)
    sdk = PayloadStore()
    restored = MooncakeRolloutArtifactStore(namespace="fixture/fresh-recovery", payload_store=sdk)
    # A new initializer's unused registration is replaced by authoritative
    # producer history. The generation is not guessed or advanced to republish.
    assert restored.register_producer("source", "unused-new-initializer") == 0
    published = await helper.publish(PublicClient(restored), saved)
    assert metadata == source_before
    assert published == metadata
    assert restored.get_current_producer_generation("source") == 1
    assert len(sdk.puts) == 1 and sdk.puts[0][1] is payload
    assert sdk.puts[0][0].startswith("fixture/fresh-recovery/")
    lease = restored.claim_group_for_artifact(
        "gate1-handoff-n8", accepted_policy_versions={"0", "1"}, lease_timeout_seconds=300,
        accepted_environment_versions={"env-fixture"}, accepted_tool_versions={"tool-fixture"},
        require_current_producer_generation=True,
    )
    assert lease.manifests == (saved.manifest,) and lease.payloads == (payload,)
    assert sdk.gets  # the native payload path completed a GET
    validate_rollout_training_payload(payload, saved.manifest,
        environment_version="env-fixture", tool_version="tool-fixture", accepted_policy_versions={"0", "1"})
    restored.acknowledge(lease)
    assert restored.get_metrics()["consumed"] == restored.get_metrics()["total"] == 1
    assert restored.claim_group_for_artifact(
        "gate1-handoff-n8", accepted_policy_versions={"0", "1"}, lease_timeout_seconds=300) is None


@pytest.mark.parametrize("status", ["writing", "leased", "consumed", "rejected"])
def test_observation_or_non_ready_record_cannot_authorize_retraining(helper, status):
    payload, metadata = saved_fixture()
    metadata["artifacts"][0]["status"] = status
    with pytest.raises(ValueError, match="authoritative READY"):
        validate(helper, payload, metadata)


def test_corrupted_payload_and_modified_generation_rejected(helper):
    payload, metadata = saved_fixture()
    with pytest.raises(ValueError, match="authoritative Artifact digest"):
        validate(helper, payload + b"tampered", metadata)
    metadata["producers"]["source"]["unexpected-C"] = 2
    with pytest.raises(ValueError, match="current-generation"):
        validate(helper, payload, metadata)
    with pytest.raises(ValueError, match="invalid schema"):
        validate(helper, payload, {"snapshots": ["observed-member-2"]})


@pytest.mark.parametrize("options, message", [
    ({"corrupt_final_reward": True}, "finite original rewards"),
    ({"placeholder": True}, "preempted placeholder"),
])
def test_incomplete_placeholder_never_becomes_a_saved_complete_group(helper, options, message):
    payload, metadata = saved_fixture(**options)
    with pytest.raises(ValueError, match=message):
        validate(helper, payload, metadata)


@pytest.mark.asyncio
async def test_saved_recovery_cannot_replace_existing_artifacts(helper):
    payload, metadata = saved_fixture()
    saved = validate(helper, payload, metadata)
    sdk = PayloadStore()
    destination = MooncakeRolloutArtifactStore(namespace="fixture/destination", payload_store=sdk)
    client = PublicClient(destination)
    await helper.publish(client, saved)
    before = client.store.export_metadata_snapshot()
    with pytest.raises(ValueError, match="fresh Artifact actor"):
        await helper.publish(client, saved)
    assert client.store.export_metadata_snapshot() == before and len(sdk.puts) == 1


@pytest.mark.asyncio
async def test_trainer_only_never_regenerates_or_changes_saved_bytes(helper, tmp_path, monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "project-b-final-validation-20260928/terminal-20261007"
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setenv("ROLLOUT_NAME", "sglang")
    import gate1_cluster_harness as harness
    from omegaconf import OmegaConf

    payload, metadata = saved_fixture()
    saved = validate(helper, payload, metadata)
    metadata_bytes = json.dumps(metadata).encode()
    sdk = PayloadStore()
    store = MooncakeRolloutArtifactStore(namespace="fixture/fresh", payload_store=sdk)
    observed_class = harness.GateObserver.__ray_metadata__.modified_class
    observer = observed_class(tmp_path, hashlib.sha256(payload).hexdigest())

    class Remote:
        def __init__(self, callback):
            self.callback = callback

        async def remote(self, *args, **kwargs):
            result = self.callback(*args, **kwargs)
            return await result if hasattr(result, "__await__") else result

    observed_handle = SimpleNamespace(
        state=Remote(observer.state), event=Remote(observer.event),
        record_actual_claim=Remote(observer.record_actual_claim),
    )

    class ObservedClient(PublicClient):
        async def claim_group_for_artifact(self, artifact_id, **kwargs):
            lease = self.store.claim_group_for_artifact(artifact_id, **kwargs)
            if lease is not None:
                observer.record_actual_claim("handoff", lease.payloads[0])
            observer.event("handoff", "claim_group_for_artifact", claimed=lease is not None)
            return lease

        async def acknowledge(self, lease):
            self.store.acknowledge(lease)
            observer.event("handoff", "acknowledge")

    client = ObservedClient(store)
    updates, notifications = [], []

    async def update(callback_client):
        updates.append(callback_client)
        lease = await callback_client.claim_group_for_artifact(
            "gate1-handoff-n8", accepted_policy_versions={"0", "1"}, lease_timeout_seconds=300)
        await callback_client.acknowledge(lease)
        return {"verification_scope": "CPU_fixture_only_no_optimizer"}

    async def notify(*, sample):
        notifications.append(sample)
        return True

    monkeypatch.setattr(harness, "GateObserver", SimpleNamespace(remote=lambda *args: observed_handle))
    monkeypatch.setattr(harness, "ObservedArtifactClient", lambda *args: client)
    components = {
        # No generation or source-resume methods exist in this control fixture.
        "rollouter": SimpleNamespace(gate_policy_identity=Remote(lambda: saved.manifest.identity)),
        "trainer": SimpleNamespace(gate_one_update=Remote(update)),
        "message_queue_client": SimpleNamespace(get_queue_size=lambda: empty_queue(), put_sample=notify),
        "artifact_store": SimpleNamespace(get_status=Remote(store.get_status), get_metrics=Remote(store.get_metrics)),
    }
    config = OmegaConf.create({"async_training": {"artifact_data_plane": {
        "environment_version": "env-fixture", "tool_version": "tool-fixture", "accepted_policy_lag": 1}}})
    result = await harness.execute_saved_trainer_gate(
        components, tmp_path, config, saved_payload=payload, saved_metadata_bytes=metadata_bytes)
    assert len(updates) == len(notifications) == 1
    assert result["executed_arms"] == [] and result["new_handoff_transactions"] == 0
    assert result["rollout_generation_calls"] == 0 and result["PREEMPTIBLE_HANDOFF"] == "NOT_RERUN"
    assert result["TRAINER_ONLY"] == result["RESTORED_GROUP_TRAINER_UPDATE"] == "PASS"
    assert (tmp_path / "imported-trainer.payload").read_bytes() == payload
    assert (tmp_path / "imported-ready-metadata.json").read_bytes() == metadata_bytes
    assert (tmp_path / "handoff-actual-trainer-claim.payload").read_bytes() == payload
    assert len(sdk.puts) == 1


async def empty_queue():
    return 0
