"""Observe the inherited trainer; never replace its update or publication."""

import hashlib
import json
import math


def learning_signal_summary(data, keys, *, group_size, group_count):
    """Observe original final-session rewards and already computed advantages."""
    import torch

    if len(keys) != len(set(keys)):
        raise ValueError("A native training output cannot appear twice in the observed batch")
    mask, advantages = data["response_mask"], data["advantages"]
    if mask.shape != advantages.shape or mask.ndim != 2 or mask.shape[0] != len(keys):
        raise ValueError("Original response masks and advantages are not aligned")
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Original response mask is not a finite binary mask")
    selected = advantages[mask.bool()]
    if selected.numel() == 0 or not torch.isfinite(selected).all():
        raise ValueError("Original effective-token advantages are empty or nonfinite")
    uids = list(data["uid"])
    final = {}
    for row, key in enumerate(keys):
        fields = key.rsplit("_", 2)  # Same final-session selection as the native GRPO helper.
        if len(fields) != 3:
            raise ValueError("Unexpected native session key")
        session, index = (fields[0], fields[1]), int(fields[2])
        if session not in final or final[session][0] < index:
            final[session] = (index, row)
    groups = {}
    for _, row in final.values():
        uid = json.dumps(uids[row], sort_keys=True)
        groups.setdefault(uid, []).append(row)
    if len(groups) != group_count or any(len(rows) != group_size for rows in groups.values()):
        raise ValueError("Training requires the expected complete eight-member prompt groups")
    scores = data["rm_scores"].sum(dim=-1)
    if not torch.isfinite(scores).all():
        raise ValueError("Original rewards are nonfinite")
    group_rows = []
    for rows in groups.values():
        digests = set()
        for row in rows:
            tokens = data["responses"][row][mask[row].bool()].tolist()
            digests.add(hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
        group_rows.append({"members": len(rows), "final_response_unique_count": len(digests),
                           "reward_std": float(scores[rows].double().std(unbiased=False))})
    nonzero = int(torch.count_nonzero(selected))
    return {"complete_prompt_groups": len(groups), "groups": group_rows,
            "effective_tokens": selected.numel(), "nonzero_advantage_tokens": nonzero,
            "nonzero_advantage_fraction": nonzero / selected.numel(),
            "advantage_l2": float(torch.linalg.vector_norm(selected.double())),
            "nonzero_training_advantage": nonzero > 0,
            "batch_keys_sha256": hashlib.sha256(json.dumps(sorted(keys)).encode()).hexdigest()}


def rank_map(rows, rank_count):
    if not isinstance(rows, list) or len(rows) != rank_count:
        raise ValueError("Every original training rank must return an observation")
    if any(type(row.get("rank")) is not int for row in rows):
        raise ValueError("Unknown training rank")
    result = {row["rank"]: row for row in rows}
    if set(result) != set(range(rank_count)):
        raise ValueError("Missing or duplicate training rank")
    return result


def verify_rank_update(before, after, *, rank_count, step):
    old, new = rank_map(before, rank_count), rank_map(after, rank_count)
    for rank, row in new.items():
        previous = old[rank]
        if row.get("trainer_step") != step or row.get("pid") != previous.get("pid"):
            raise ValueError("The original training process or step changed")
        if row["optimizer_calls_since_attach"] - previous["optimizer_calls_since_attach"] != 1:
            raise ValueError("Expected exactly one actual optimizer call on every rank")
        if (
            row["verified_counter_advancements_since_attach"] - previous["verified_counter_advancements_since_attach"]
            != 1
        ):
            raise ValueError("Optimizer counter advancement was not verified on every rank")
    return {"trainer_step": step, "rank_count": rank_count, "verified_updates_per_rank": 1}


def verify_publication(observation, step, model_configuration_fingerprint):
    if type(observation.get("actual_parameter_version")) is not int:
        raise ValueError("Actual engine parameter version is unknown")
    if observation["actual_parameter_version"] != step:
        raise ValueError("Native engines did not confirm the updated parameter version")
    record = observation["active_policy_record"]
    if record["identity"]["policy_version"] != str(step):
        raise ValueError("Durable policy record is stale")
    if record["model_configuration_fingerprint"] != model_configuration_fingerprint:
        raise ValueError("Model configuration changed across publication")


def make_audited_trainer(
    base, *, rpc, event, read_publication, rank_count, model_configuration_fingerprint, begin_audit, read_audit,
    read_learning_signal=None,
):
    """Keep stock fit/step/checkpoint and instrument their native call sites."""

    class AuditedTrainer(base):
        def _compute_advantage(self, batch, metrics):
            result = super()._compute_advantage(batch, metrics)
            if read_learning_signal is not None:
                signal = read_learning_signal(self, result)
                if not hasattr(self, "audit_learning_signals"):
                    self.audit_learning_signals = {}
                self.audit_learning_signals[self.global_steps] = signal
                event("native_learning_signal_observed", trainer_step=self.global_steps, **signal)
            return result

        def _update_actor(self, batch, metrics):
            step = self.global_steps
            if step in self.audit_update_steps:
                raise RuntimeError("Never repeat an optimizer update to repair its evidence")
            signal = getattr(self, "audit_learning_signals", {}).get(step)
            if read_learning_signal is not None and signal is None:
                raise ValueError("Original computed advantage must be observed before dispatch")
            if signal is not None:
                keys = hashlib.sha256(json.dumps(sorted(batch.keys)).encode()).hexdigest()
                if keys != signal["batch_keys_sha256"]:
                    raise ValueError("Observed learning batch differs from the dispatched original batch")
            before = rpc(self, read_audit)
            rpc(self, begin_audit, step)
            self.audit_update_steps.append(step)  # Mark dispatched before stock training.
            event("native_actor_update_dispatched", trainer_step=step, observations=before)
            try:
                result = super()._update_actor(batch, metrics)
            except BaseException:
                event(
                    "native_actor_update_failed",
                    trainer_step=step,
                    optimizer_outcome="requires_readback",
                    automatic_update_retry_allowed=False,
                )
                raise
            after = rpc(self, read_audit)
            proof = verify_rank_update(before, after, rank_count=rank_count, step=step)
            gradient = metrics.get("actor/grad_norm")
            gradient = float(gradient) if isinstance(gradient, (int, float)) else None
            if gradient is not None and not math.isfinite(gradient):
                raise ValueError("The original actor returned a nonfinite gradient norm")
            policy_loss = metrics.get("actor/pg_loss")
            policy_loss = float(policy_loss) if isinstance(policy_loss, (int, float)) else None
            if policy_loss is not None and not math.isfinite(policy_loss):
                raise ValueError("The original actor returned a nonfinite policy loss")
            learning = (signal is not None and signal["nonzero_training_advantage"]
                        and gradient is not None and gradient > 0 and policy_loss is not None)
            if learning:
                if not hasattr(self, "audit_learning_update_steps"):
                    self.audit_learning_update_steps = []
                self.audit_learning_update_steps.append(step)
            self.audit_verified_steps.append(step)
            event(
                "native_actor_update_verified",
                **proof,
                observations=after,
                gradient_norm=gradient,
                policy_loss=policy_loss,
                learning_signal_verified=learning,
                parameter_change_verified=None,
            )
            return result

        def on_step_end(self):
            result = super().on_step_end()
            proof = read_publication(self)
            verify_publication(proof, self.global_steps, model_configuration_fingerprint)
            self.audit_publication_steps.append(self.global_steps)
            event(
                "native_policy_publication_verified",
                trainer_step=self.global_steps,
                **proof,
                controller_reclaims=list(getattr(self, "trajectory_resource_reclaims", [])),
                physical_gpu_release_verified=False,
            )
            from os import environ

            if environ.get("VERL_STORAGE_HANDOFF_AT_STEP") or environ.get("VERL_STORAGE_HANDOFF_CAMPAIGN_SPEC"):
                from storage_handoff_driver import maybe_run_after_publication

                maybe_run_after_publication(self, proof)
            return result

        def _save_checkpoint(self):
            event("native_checkpoint_save_dispatched", trainer_step=self.global_steps)
            result = super()._save_checkpoint()
            event(
                "native_checkpoint_save_returned",
                trainer_step=self.global_steps,
                full_restore_equivalence_verified=False,
            )
            return result

    return AuditedTrainer
