"""Ray store-actor integration for preemption + handoff manifests (M1 wiring)."""
from __future__ import annotations

from dataclasses import asdict

import json
import os
import asyncio
import pytest

import ray

from verl.experimental.fully_async_policy import rollout_artifact as ray_artifact
from verl.experimental.fully_async_policy.preemptible_handoff import HandoffManifest, HandoffState, PreemptState
from verl.experimental.fully_async_policy.rollout_artifact import CanonicalPolicyKVIdentity

PRODUCER = "fully_async_rollouter"


def member(index, group="group-1", size=8, generation=0):
    return {"trajectory_id": f"{group}#{index}", "group_id": group,
            "member_index": index, "group_size": size}


def build_manifest(group="group-1", completed=(0, 1), checkpointed=(2, 3, 4, 5), pending=(6, 7)):
    return HandoffManifest(
        handoff_id="handoff-1",
        source_rollouter=PRODUCER,
        source_generation=ray.get(store.get_current_producer_generation.remote(PRODUCER)),
        policy_identity=asdict(CanonicalPolicyKVIdentity("0", "weights", "template", "kv")),
        environment_version="env-1",
        tool_version="tools-1",
        completed_artifacts=(),
        checkpointed_trajectories=tuple({
            **member(index, group), "expected_generation": 0, "snapshot_ref": f"inflight/snap-{index}",
        } for index in checkpointed),
        completed_members=tuple({
            **member(index, group), "record_ref": f"handoff-member/r{index}",
        } for index in completed),
        pending_samples=tuple(member(index, group) for index in pending),
    )


def init_ray_store():
    ray.init(num_cpus=1, include_dashboard=False, log_to_driver=False)
    StoreActor = ray_artifact.create_ray_rollout_artifact_store_actor(trajectory_snapshot_enabled=True)
    store = StoreActor.remote()
    ray.get(store.register_producer.remote(PRODUCER, "incarnation-1"))
    return store


def get(obj):
    return ray.get(obj)


def test_store_actor_preemption_and_manifest_transaction():
    global store
    try:
        store = init_ray_store()

        # --- cooperative preemption states over the actor boundary ---
        get(store.register_running_trajectory.remote("group-1#2"))
        assert get(store.get_trajectory_preemption.remote("group-1#2")) == "running"
        assert get(store.request_trajectory_preemption.remote("group-1#2")) == "preempt_requested"
        assert get(store.mark_trajectory_safe_point.remote("group-1#2")) == "safe_point_reached"
        assert get(store.mark_trajectory_snapshot_committed.remote(
            "group-1#2", "inflight/snap-2")) == "snapshot_committed"
        assert get(store.mark_trajectory_handed_off.remote("group-1#2")) == "handed_off"
        assert get(store.get_trajectory_preemption_metrics.remote())["trajectories"][0][
            "trajectory_id"] == "group-1#2"

        # --- manifest fails closed while a listed snapshot is missing ---
        draft = build_manifest()
        result = get(store.prepare_handoff_manifest.remote(draft))
        assert result == {"handoff_id": "handoff-1", "state": "preparing"}
        try:
            get(store.commit_handoff_manifest.remote("handoff-1"))
            raise AssertionError("commit must fail before every snapshot is committed")
        except ValueError:
            pass

        # --- the remaining listed trajectories reach their safe points ---
        for index in (3, 4, 5):
            trajectory = f"group-1#{index}"
            get(store.register_running_trajectory.remote(trajectory))
            get(store.request_trajectory_preemption.remote(trajectory))
            get(store.mark_trajectory_safe_point.remote(trajectory))
            get(store.mark_trajectory_snapshot_committed.remote(trajectory, f"inflight/snap-{index}"))
        assert get(store.commit_handoff_manifest.remote("handoff-1")) == "committed"
        assert get(store.record_handoff_source_release.remote("handoff-1")) == "committed"
        assert get(store.begin_handoff_restore.remote("handoff-1")) == "restoring"
        try:
            get(store.finish_handoff_manifest.remote("handoff-1", "unpublished"))
            raise AssertionError("finish must reject an unpublished/incomplete group")
        except ValueError:
            pass
        metrics = get(store.get_handoff_metrics.remote())["handoffs"][0]
        assert metrics["state"] == "restoring"
        assert metrics["worker_reclaim_latency_ms"] is not None

        # --- a stale source generation fails closed at prepare time ---
        stale = build_manifest()
        object.__setattr__(stale, "source_generation", stale.source_generation + 1)
        try:
            get(store.prepare_handoff_manifest.remote(stale))
            raise AssertionError("stale generation must fail closed")
        except ValueError:
            pass

        # --- an unknown producer also fails closed ---
        ghost = build_manifest()
        ghost = HandoffManifest(**{**asdict(ghost), "handoff_id": "handoff-ghost",
                                   "source_rollouter": "ghost_rollouter", "source_generation": 0})
        try:
            get(store.prepare_handoff_manifest.remote(ghost))
            raise AssertionError("unknown producer must fail closed")
        except ValueError:
            pass

        # --- member record payloads round-trip through the actor ---
        import json as _json
        reference = get(store.put_handoff_member_record.remote(
            _json.dumps({"member_index": 0}).encode()))
        payload = get(store.get_handoff_member_record.remote(reference))
        assert _json.loads(payload)["member_index"] == 0
        get(store.discard_handoff_member_record.remote(reference))
    finally:
        ray.shutdown()


def test_async_client_trajectory_rpc_preserves_real_actor_fencing():
    from tests.experimental.fully_async_policy.test_trajectory_snapshot import snapshot

    async def exercise(actor):
        client = ray_artifact.RayRolloutArtifactStoreClient(actor)
        a = await client.open_trajectory("task-1", "source", 60)
        await client.check_trajectory_owner(a)
        a = await client.renew_trajectory_owner(a, 60)
        await client.commit_trajectory_snapshot(a, snapshot(a.generation), 300)
        assert await client.load_trajectory_snapshot(a) == snapshot(a.generation)
        await client.release_trajectory_owner(a)
        b = await client.takeover_trajectory("task-1", a.generation, "target", 60)
        assert b.generation == a.generation + 1
        with pytest.raises(ValueError, match="stale"):
            await client.check_trajectory_owner(a)
        with pytest.raises(ValueError, match="stale"):
            await client.commit_trajectory_snapshot(a, snapshot(a.generation), 300)
        await client.finish_trajectory(b)
        assert await client.collect_expired_trajectory_snapshots() == 0

    try:
        actor = init_ray_store()
        asyncio.run(exercise(actor))
    finally:
        ray.shutdown()


def main():
    assert HandoffState.PREPARING.value == "preparing"
    assert PreemptState.SAFE_POINT_REACHED.value == "safe_point_reached"
    test_store_actor_preemption_and_manifest_transaction()
    print("STORE_ACTOR_WIRING_OK")


if __name__ == "__main__":
    main()
