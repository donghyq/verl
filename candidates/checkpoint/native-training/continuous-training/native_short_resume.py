"""Read-only evidence around stock fit and stock checkpoint calls (3 -> 4)."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def observe_native_progress(worker):
    """Use the existing shard probe; read the actual scheduler, process and Adam."""
    import ray
    from actor_worker import observe_original_parameters, resolve_training_worker

    row = observe_original_parameters(worker)
    selected = resolve_training_worker(worker)
    engine = selected.actor.engine if hasattr(selected, "actor") else selected.engine
    scheduler = engine.lr_scheduler
    if scheduler is None:
        raise ValueError("Native scheduler evidence is required")
    row["scheduler_state"] = scheduler.state_dict()
    context = ray.get_runtime_context()
    row["ray_actor_id"] = str(context.get_actor_id())
    row["ray_node_id"] = str(context.get_node_id())
    row["hostname"] = socket.gethostname()
    stat = Path("/proc/self/stat")
    row["linux_process_start_ticks"] = int(stat.read_text().rpartition(") ")[2].split()[19]) if stat.is_file() else None
    json.dumps(row, allow_nan=False)
    if row.get("error_type") or not row["ray_actor_id"]:
        raise ValueError("Original native parameter or actor observation failed")
    return row


def verify_restored_progress(before, prior, *, step, rank_count=4):
    from native_training_audit import rank_map
    from run_v1_functional_gate import parameter_sample_delta

    left, right = rank_map(prior, rank_count), rank_map(before, rank_count)
    proof = parameter_sample_delta(prior, before)
    if proof.get("sample_linf") != 0.0:
        raise ValueError("Restored original shard samples differ from the saved process")
    ranks = []
    for rank, row in right.items():
        previous = left[rank]
        if row.get("optimizer_step_max") != step or row.get("scheduler_state") != previous.get("scheduler_state"):
            raise ValueError("Native Adam or scheduler did not restore the saved frontier")
        same_process = all(row.get(key) == previous.get(key)
                           for key in ("hostname", "pid", "linux_process_start_ticks"))
        if not row.get("ray_actor_id") or same_process:
            raise ValueError("Resume requires a newly initialized native training actor")
        ranks.append({"rank": rank, "previous_actor_id": previous["ray_actor_id"],
                      "restored_actor_id": row["ray_actor_id"], "optimizer_step": step,
                      "previous_pid": previous["pid"], "restored_pid": row["pid"],
                      "scheduler_exact_state_match": True})
    return {"restored_global_step": step, "parameter_samples": proof, "ranks": ranks,
            "new_native_training_actors": True, "full_restore_equivalence_verified": False}


def checkpoint_inventory(checkpoint, *, rank_count=4):
    """Check the existing native format without hashing/deserializing large shards."""
    root = Path(checkpoint)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing native checkpoint directory")
    rows = []
    for rank in range(rank_count):
        for prefix in ("model", "optim", "extra_state"):
            relative = f"actor/{prefix}_world_size_{rank_count}_rank_{rank}.pt"
            path = root / relative
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise ValueError("Missing native checkpoint shard: " + relative)
            rows.append({"path": relative, "bytes": path.stat().st_size})
    for name in ("data.pt", "transfer_queue", "trajectory_state"):
        path = root / name
        if path.is_symlink() or not path.exists() or (path.is_file() and not path.stat().st_size):
            raise ValueError("Missing native progress component: " + name)
        if path.is_dir() and not any(child.is_file() and child.stat().st_size for child in path.rglob("*")):
            raise ValueError("Empty native progress component: " + name)
    return {"path": str(root), "shards": rows, "world_size": rank_count,
            "contents": ["model", "optimizer", "scheduler_and_rng", "dataloader", "transfer_queue", "trajectory_state"],
            "full_shard_sha256_performed": False, "checkpoint_deserialized_by_observer": False}


def checkpoint_prefetch_provenance():
    """Read actual saved prompt identities; no queue writes or synthetic dispatch."""
    import transfer_queue as tq
    from runtime_observer import fingerprint, plain

    rows = (tq.kv_list("train") or {}).get("train", {})
    uids = sorted(uid for uid, tag in rows.items() if tag.get("is_prompt") is True)
    if not uids:
        return {}
    batch = tq.kv_batch_get(keys=uids, partition_id="train", select_fields=["raw_prompt"])
    return {uid: {"initial_raw_prompt_sha256": fingerprint(plain(batch[index]["raw_prompt"])),
                  "saved_prompt_tag": rows[uid]} for index, uid in enumerate(uids)}


def verify_restored_prefetch(prior_directory):
    saved = json.loads((Path(prior_directory) / "checkpoint-prefetch-provenance.json").read_text())
    actual = checkpoint_prefetch_provenance()
    if set(actual) != set(saved) or any(actual[uid] != saved[uid] for uid in actual):
        raise ValueError("Restored TransferQueue prompt identity or tag differs from native checkpoint")
    return {uid: {"initial_raw_prompt_sha256": row["initial_raw_prompt_sha256"],
                  "monotonic_ns": None, "utc_unix_s": None,
                  "origin": "actual_native_checkpoint_prefetch_not_new_process_dispatch",
                  "native_staleness_and_version_validation_unchanged": True}
            for uid, row in actual.items()}


def instrument_short_trainer(base, *, rpc, event, output, prior_directory=None, checkpoint_guard=None):
    """Observe every actual batch/update. No sampled batch or optimizer is replaced."""
    from native_training_audit import rank_map
    from run_v1_functional_gate import parameter_sample_delta
    from runtime_observer import group_membership, input_observation

    directory = Path(output)
    seen = set()
    if prior_directory is not None:
        for path in sorted(Path(prior_directory).glob("step-*/batch.json")):
            seen.update(json.loads(path.read_text())["group_ids"])
        if not seen:
            raise ValueError("Resume requires the previous process consumed-group records")

    class ShortNativeTrainer(base):
        def _save_checkpoint(self):
            from native_checkpoint_guard import (require_save_budget, append_event,
                                                  complete_native_save, observe_bound_method)
            started = time.monotonic()
            step = self.global_steps
            checkpoint = Path(self.config.trainer.default_local_dir) / f"global_step_{step}"
            log = directory.parent / "native-checkpoint-stages.jsonl"
            budget = require_save_budget(checkpoint_guard)
            append_event(log, status="SAVE_BUDGET_ACCEPTED", step=step, **budget)
            try:
                import contextlib
                with contextlib.ExitStack() as stack:
                    manager = self.agent_loop_manager
                    for method, stage in (("prepare_trajectory_checkpoint", "trajectory_quiesce"),
                            ("export_trajectory_checkpoint", "trajectory_export"),
                            ("resume_trajectory_checkpoint_admission", "trajectory_resume_admission")):
                        stack.enter_context(observe_bound_method(manager, method, log, stage, step=step))
                    result = super()._save_checkpoint()
                proof = complete_native_save(checkpoint, step,
                    receipt_directory=directory.parent/"native-checkpoint-rank-receipts",
                    elapsed_s=time.monotonic()-started)
            except BaseException as error:
                append_event(log, status="SAVE_INCOMPLETE", step=step,
                    error_type=type(error).__name__, elapsed_s=time.monotonic()-started)
                raise
            append_event(log, status=proof["status"], step=step,
                elapsed_s=time.monotonic()-started)
            return result

        def _save_checkpoint_native(self):
            result = super()._save_checkpoint_native()
            # The stock parent _save_checkpoint has paused trajectory admission
            # and has not reached its finally/resume clause at this point.
            save_json(directory.parent / "checkpoint-prefetch-provenance.json", checkpoint_prefetch_provenance())
            return result

        def _update_actor(self, batch, metrics):
            step = self.global_steps
            step_dir = directory / f"step-{step}"
            groups = group_membership(batch.keys)
            if len(groups) != 8 or seen.intersection(groups):
                raise ValueError("Require eight complete previously unused native prompt-group UIDs")
            metadata = input_observation(batch, {})
            # This observer has no dispatch table. Do not call that absence a mismatch.
            metadata["dispatch_match_checked_here"] = False
            metadata["repeated_consumed_group_rejected"] = True
            metadata["group_ids"] = sorted(groups)
            save_json(step_dir / "batch.json", metadata)
            if metadata.get("status") != "OBSERVED_ACTUAL_GROUP_METADATA":
                raise ValueError("Actual business tokens, masks and behavior versions must be observable before dispatch")
            for group in metadata["groups"]:
                for member in group["members"]:
                    if (not member["behavior_logprob_aligned_finite"]
                            or member["training_effective_model_tokens"] <= 0
                            or not member.get("behavior_segments")
                            or any(segment.get("policy_version") is None for segment in member["behavior_segments"])):
                        raise ValueError("Missing real effective tokens or native behavior-version metadata")
            seen.update(groups)  # Once dispatched, never retry this update for evidence.
            before = rpc(self, observe_native_progress)
            rank_map(before, 4)
            save_json(step_dir / "parameters-before.json", before)
            result = super()._update_actor(batch, metrics)
            after = rpc(self, observe_native_progress)
            old, new = rank_map(before, 4), rank_map(after, 4)
            proofs = []
            for rank in range(4):
                proof = parameter_sample_delta([old[rank]], [new[rank]])
                proofs.append({"rank": rank, **proof})
                if proof.get("observed_parameter_change") is not True or new[rank]["optimizer_step_max"] != step:
                    save_json(step_dir / "parameters-after.json", after)
                    save_json(step_dir / "shard-deltas.json", proofs)
                    raise ValueError("An actual native rank lacks the required parameter change/Adam frontier")
            save_json(step_dir / "parameters-after.json", after)
            save_json(step_dir / "shard-deltas.json", proofs)
            event("short_native_all_rank_shard_changes", trainer_step=step, ranks=proofs)
            return result

    return ShortNativeTrainer
