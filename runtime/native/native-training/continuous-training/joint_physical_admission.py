"""Necessary physical admission before an original target dispatch; native restore still verifies tensors."""
import hashlib
import json
from pathlib import Path

async def require_original_physical_handoff(client, *, handoff_id, uid, expected):
    from verl.trainer.ppo.v1.trajectory_state.snapshot import TrajectorySnapshot
    from verl.trainer.ppo.v1.trajectory_state.token_kv_resume import read_committed_request
    manifest = await client.get_handoff_manifest(handoff_id)
    members = manifest["checkpointed_trajectories"] if isinstance(manifest, dict) else manifest.checkpointed_trajectories
    observed = []
    for member in members:
        raw = await client.get_handoff_member_record(member["snapshot_ref"])
        snapshot = TrajectorySnapshot.decode(raw)
        ref = snapshot.kv_ref
        observed.append({"trajectory_id": snapshot.trajectory_id, "schema_version": snapshot.schema_version,
            "kv_kind": (ref or {}).get("kind")})
        bindings = expected.get("active_requests", {expected["trajectory_id"]: expected})
        active = bindings.get(snapshot.trajectory_id)
        if active is None or snapshot.schema_version != 2 or ref is None:
            continue  # Other original schema1/completed/pending members retain their legal paths.
        if (snapshot.group_id != uid or snapshot.member_index != active["member_index"]
                or snapshot.execution_generation != active["source_execution_generation"]
                or not isinstance(ref, dict)
                or ref.get("kind") != "token_kv_committed_v1" or ref.get("cache_level", "PORTABLE") != "PORTABLE"
                or ref.get("snapshot_committed") is not True
                or ref.get("identity", {}).get("tp_size") != 2):
            raise ValueError("Selected active original member is not a committed schema2 physical checkpoint: " + json.dumps(observed))
        binding = ref.get("source_binding", {})
        for key in ("trajectory_id", "source_execution_generation", "source_owner_id", "source_lease_id", "source_engine_instance_id", "request_id"):
            if binding.get(key) != active[key]:
                raise ValueError("Physical checkpoint lost original live request/owner binding: " + key)
        if (str(ref["identity"].get("weight_version")) != str(active["engine_weight_version"])
                or str(ref["identity"].get("actualweight_version")) != str(active["engine_weight_version"])):
            raise ValueError("Physical checkpoint engine version differs from observed live request")
        # Existing reader checks immutable COMMITTED.json and every TP metadata/shared-state hash.
        request = read_committed_request(ref)
        if len(request.get("output_ids", [])) < 18:
            raise ValueError("Native committed generated prefix is below the original 18-token threshold")
        directory = Path(ref["manifest_ref"]["path"]).parent
        ranks = []
        for rank in range(2):
            metadata = json.loads((directory / ("rank-%d.json" % rank)).read_bytes())
            payload = directory / ("rank-%d.bin" % rank)
            if payload.is_symlink() or not payload.is_file() or payload.stat().st_size != metadata["payload_bytes"]:
                raise ValueError("Physical TP payload absent or changed before target dispatch")
            digest = hashlib.sha256()
            with payload.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != metadata["payload_sha256"]:
                raise ValueError("Physical TP payload hash differs before target dispatch")
            ranks.append({"rank": rank, "payload_bytes": metadata["payload_bytes"], "payload_sha256": digest.hexdigest()})
        return {"status": "COMMITTED_PHYSICAL_REQUIRED_GATE_PASSED", "uid": uid, "handoff_id": handoff_id,
            "trajectory_id": snapshot.trajectory_id, "schema_version": 2, "native_request_id": binding["request_id"],
            "ranks": ranks, "native_target_tensor_restore_still_required": True}
    raise ValueError("Selected active member has no physical snapshot before target dispatch: " + json.dumps(observed))
