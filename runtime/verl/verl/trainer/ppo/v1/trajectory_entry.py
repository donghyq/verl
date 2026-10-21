# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Opt-in binding from the stock V1 worker to tool-boundary state handoff.

The existing tool handlers, reward processing, queue schema and Trainer remain
authoritative. A driver chooses the target worker explicitly; this is not a
resource scheduler. Uncertain publication never falls back to fresh generation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import NonTensorData, NonTensorStack

from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap


@dataclass(frozen=True)
class PreparedTrajectoryMember:
    """The original postprocessed result, retained before any queue dispatch."""

    output: object
    keys: list[str]
    fields: object
    tags: list[dict]


class V1TrajectoryEntry:
    def __init__(self, worker, runtime):
        self.worker, self.runtime = worker, runtime
        # Populated only by the controlled acceptance fixture. Ordinary
        # training does not retain extra snapshots or expose owner credentials.
        self._acceptance_source_records = {}
        self._acceptance_boundaries = {}

    def enable_acceptance_source_observation(self, uid, *, checkpoint_members=None):
        if uid in self._acceptance_source_records:
            raise ValueError("acceptance source observation is already enabled for this prompt")
        if checkpoint_members is not None:
            if (not isinstance(checkpoint_members, list) or not checkpoint_members
                    or any(type(i) is not int or not 0 <= i < 8 for i in checkpoint_members)
                    or len(set(checkpoint_members)) != len(checkpoint_members)):
                raise ValueError("acceptance checkpoint members must be distinct valid n=8 indices")
            self._acceptance_boundaries[uid] = {
                "members": frozenset(checkpoint_members), "ready": set(),
                "release": asyncio.Event(),
            }
        self._acceptance_source_records[uid] = {}

    def acceptance_boundary_status(self, uid):
        state = self._acceptance_boundaries.get(uid)
        if state is None:
            return {}
        return {
            "acceptance_boundary_members": sorted(state["members"]),
            "acceptance_boundary_ready_members": sorted(state["ready"]),
            "acceptance_boundary_released": state["release"].is_set(),
        }

    def release_acceptance_boundaries(self, uid):
        state = self._acceptance_boundaries.get(uid)
        if state is not None:
            state["release"].set()

    def acceptance_source_records(self, uid):
        """Read actual committed source objects for a controlled stale-write probe."""
        records = self._acceptance_source_records.get(uid)
        return deepcopy(records) if records is not None else None

    def _new_loop(self, original):
        from .trajectory_state.preemptible_tool_agent_loop import DurableToolAgentLoop

        kwargs = dict(
            trainer_config=DictConfigWrap(config=self.worker.config),
            server_manager=self.worker.llm_client,
            tokenizer=self.worker.tokenizer,
            processor=self.worker.processor,
            hf_model_type=self.worker.hf_model_type,
            dataset_cls=self.worker.dataset_cls,
            data_config=DictConfigWrap(self.worker.config.data),
            tools=ToolListWrap(self.worker.tools),
        )
        agent_name = original.get("agent_name")
        if agent_name in {"tool_agent", "durable_tool_agent"}:
            return DurableToolAgentLoop(**kwargs)

        import hydra
        from verl.experimental.agent_loop.agent_loop import _agent_loop_registry

        if agent_name not in _agent_loop_registry:
            raise ValueError(f"durable agent loop is not registered: {agent_name}")
        loop = hydra.utils.instantiate(config=_agent_loop_registry[agent_name], **kwargs)
        if not isinstance(loop, DurableToolAgentLoop):
            raise ValueError("durable agent loop must inherit DurableToolAgentLoop")
        return loop

    @staticmethod
    def _prompt_kwargs(original):
        return {key: deepcopy(value) for key, value in original.items()
                if key not in {"sampling_params", "trajectory", "trace", "agent_name"}}

    def _identity_kwargs(self, i, original, lease):
        return {
            "trajectory_id": lease.trajectory_id,
            "rollout_group_id": original["uid"],
            "rollout_sample_index": i,
            "rollout_group_size": original["__trajectory_group_size__"],
            "execution_generation": lease.generation,
            "snapshot_policy_identity": original.get(
                "__trajectory_snapshot_policy_identity__", self.runtime.policy_identity
            ),
            "snapshot_environment_version": self.runtime.environment_version,
            "snapshot_tool_version": self.runtime.tool_version,
        }

    async def _run_source_member(self, i, original, lease, commit_snapshot, check_owner):
        loop = self._new_loop(original)
        kwargs = self._prompt_kwargs(original)
        kwargs.pop("__trajectory_group_size__", None)
        kwargs.update(self._identity_kwargs(i, original, lease))

        if getattr(self.worker, "token_kv_source_backend", None) is not None:
            kwargs["_trajectory_token_kv_source_backend"] = self.worker.token_kv_source_backend
            kwargs["_trajectory_source_lease"] = lease

        boundary = self._acceptance_boundaries.get(original["uid"])
        if boundary is not None and i in boundary["members"]:
            async def wait_at_completed_tool_batch(turn_index, batch_count):
                if type(turn_index) is not int or turn_index < 0 or batch_count < 1:
                    raise ValueError("acceptance requires a completed real tool batch")
                boundary["ready"].add(i)
                await boundary["release"].wait()
            kwargs["_trajectory_acceptance_tool_boundary"] = wait_at_completed_tool_batch

        async def observe_committed_snapshot(snapshot):
            reference = await commit_snapshot(snapshot)
            records = self._acceptance_source_records.get(original["uid"])
            if records is not None and reference is not None:
                records[i] = deepcopy({
                    "owner_lease": lease, "snapshot": snapshot, "snapshot_ref": reference,
                })
            return reference

        for name in ("resolve_published_policy", "get_snapshot_policy_identity"):
            if hasattr(commit_snapshot, name):
                setattr(observe_committed_snapshot, name, getattr(commit_snapshot, name))

        output = await loop.run_until_checkpoint(
            original["sampling_params"], commit_snapshot=observe_committed_snapshot,
            check_owner=check_owner, **kwargs
        )
        await check_owner()
        return await self._prepare_member(i, output, original)

    async def _run_pending_member(self, i, original, lease, check_owner):
        # Pending members execute for the first time; the callback can never
        # create another snapshot during this one target restoration.
        async def no_checkpoint(snapshot):
            return None

        return await self._run_source_member(i, original, lease, no_checkpoint, check_owner)

    async def _resume_member(self, i, snapshot, original, lease, check_owner):
        loop = self._new_loop(original)
        # Existing V1 tool-boundary snapshots and CPU test doubles may predate
        # schema v2. Only an explicit v2 snapshot can select token-KV restore.
        if getattr(snapshot, "schema_version", 1) == 2:
            from .trajectory_state.token_kv_resume import install_token_kv_resume

            install_token_kv_resume(
                loop, snapshot=snapshot, lease=lease, check_owner=check_owner,
                backend=getattr(self.worker, "token_kv_resume_backend", None),
            )
        kwargs = self._prompt_kwargs(original)
        kwargs.pop("__trajectory_group_size__", None)
        kwargs.update(self._identity_kwargs(i, original, lease))
        kwargs["snapshot_policy_identity"] = deepcopy(snapshot.policy_identity)
        # resume_from_snapshot defines these explicitly; do not pass duplicates.
        for key in ("execution_generation", "snapshot_environment_version", "snapshot_tool_version"):
            kwargs.pop(key, None)
        output = await loop.resume_from_snapshot(
            snapshot, check_owner=check_owner, execution_generation=lease.generation,
            environment_version=self.runtime.environment_version,
            tool_version=self.runtime.tool_version, **kwargs
        )
        await check_owner()
        recovery = getattr(output, "extra_fields", {}).get("trajectory_token_kv_recovery")
        receipt_dir = os.environ.get("VERL_TRAJECTORY_RECEIPT_DIR")
        if isinstance(recovery, dict) and recovery.get("mode") in {
            "PORTABLE_STATE_RESTORED", "NATIVE_TOKEN_RECOMPUTE"
        } and receipt_dir:
            marker = deepcopy(recovery)
            actual_group = self.runtime.group_status(original["uid"])
            if not actual_group or not actual_group.get("handoff_id"):
                raise ValueError("native recovery receipt requires its actual target handoff")
            marker.update({
                "handoff_id": actual_group["handoff_id"],
                "uid": original["uid"], "member_index": i,
                "trajectory_id": lease.trajectory_id,
                "source_execution_generation": getattr(snapshot, "execution_generation", None),
                "target_execution_generation": lease.generation,
            })
            name = "recovery-" + hashlib.sha256(lease.trajectory_id.encode("utf-8")).hexdigest()[:16] + ".json"
            path = Path(receipt_dir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n")
        return await self._prepare_member(i, output, original)

    async def _prepare_member(self, i, output, original):
        from .trajectory_state.handoff import save_unscored_member

        # Commit the actual generated/tool result before a fallible scorer runs.
        # This immutable record is explicitly not a prepared/trainable member.
        score_kwargs = self._postprocess_kwargs(original)
        await save_unscored_member(self.runtime.client, i, output, original, score_kwargs=score_kwargs)
        row = await self.worker._prepare_agent_loop_outputs(
            output, False, **score_kwargs
        )
        if row is None or row[0] != [f"{original['uid']}_{i}_0"]:
            raise ValueError("a trajectory must have exactly one original V1 output with its own member key")
        keys, fields, tags = row
        validate_trajectory_fields(fields)
        return PreparedTrajectoryMember(output, keys, fields, tags)

    async def run_prompt(self, prompt, sampling_params, trajectory, *, trace=False, start_members=None):
        if trajectory["validate"]:
            raise ValueError("the opt-in handoff entry covers training prompts only")
        config = self.worker.config.actor_rollout_ref.rollout
        data = deepcopy(prompt)
        n = data.pop("__rollout_n__", config.n)
        do_sample = data.pop("__do_sample__", True)
        if type(n) is not int or n != 8:
            raise ValueError("the first durable handoff entry requires exactly eight members")
        params = deepcopy(sampling_params)
        if not do_sample:
            from .agent_loop_tq import apply_greedy_sampling_params

            apply_greedy_sampling_params(params)
        members = [
            {**deepcopy(data), "session_id": i, "sampling_params": deepcopy(params),
             "trajectory": deepcopy(trajectory), "trace": trace, "__trajectory_group_size__": n}
            for i in range(n)
        ]
        result = await self.runtime.run_source(
            data["uid"], members, self._run_source_member, start_members=start_members
        )
        if result.status == "completed":
            import transfer_queue as tq

            for output in result.outputs:
                if not isinstance(output, PreparedTrajectoryMember):
                    raise ValueError("durable members must contain their original prepared V1 fields")
                await tq.async_kv_batch_put(
                    keys=output.keys, fields=output.fields, tags=output.tags, partition_id="train"
                )
        return result

    async def confirm_normal_publication(self, result):
        """Confirm a successful prompt finished PUT, then retry storage cleanup."""
        await self.runtime.confirm_normal_publication(result)

    async def cleanup_completed_group(self, result):
        """Retry cleanup for an already-published normal group without republishing."""
        return await self.runtime.cleanup_completed_group(result)

    @staticmethod
    def _postprocess_kwargs(original):
        return {key: value for key, value in original.items()
                if key not in {"sampling_params", "trajectory", "trace", "__trajectory_group_size__"}}

    async def resume_handoff(self, handoff_id):
        from .trajectory_publication import (
            PublicationMember,
            publish_trajectory_group,
            reconcile_trajectory_group_publication,
        )

        state = await self.runtime.client.summary()
        if state["handoffs"].get(handoff_id) == "finished":
            # The real queue may already have been consumed. The publication
            # adapter validates the durable ready proof without recreating keys.
            receipt = await reconcile_trajectory_group_publication(
                coordinator=self.runtime.client, handoff_id=handoff_id,
                validate_fields=validate_trajectory_fields,
            )
            self._require_finished_publication(receipt)
            return None, receipt
        publication = await self.runtime.client.get_publication(handoff_id)
        if publication is not None and publication["state"] != "PREPARED":
            # Recovery reads the existing binding and immutable prepared data;
            # it never resumes members, recomputes rewards or dispatches PUT.
            receipt = await reconcile_trajectory_group_publication(
                coordinator=self.runtime.client, handoff_id=handoff_id,
                validate_fields=validate_trajectory_fields,
            )
            self._require_finished_publication(receipt)
            return None, receipt
        target_options = {}
        policy_client = getattr(self.runtime, "published_policy_client", None)
        if policy_client is not None:
            from .trajectory_state.policy_publication import PublishedPolicyRecord

            manifest = await self.runtime.client.get_handoff_manifest(handoff_id)
            active = PublishedPolicyRecord.decode(await policy_client.get_active_published_policy())
            if manifest.policy_identity != vars(active.identity):
                # Keep historical behavior logprobs, but regenerate attention
                # state from the exact token prefix under the current weights.
                target_options["token_only"] = True
        result = await self.runtime.run_target(
            handoff_id, self._resume_member, self._run_pending_member, **target_options
        )
        prepared = []
        members = []
        tags = []
        for i, (output, original) in enumerate(zip(result.outputs, result.member_kwargs, strict=True)):
            if not isinstance(output, PreparedTrajectoryMember):
                raise ValueError("restored members must retain original reward/teacher/postprocessing fields")
            keys, fields, row_tags = output.keys, output.fields, output.tags
            if keys != [f"{original['uid']}_{i}_0"]:
                raise ValueError("restored member keys differ from the stock V1 identity")
            prepared.append(fields)
            tags.extend(row_tags)
            members.append(PublicationMember(original["uid"], i, 0))
        fields = assemble_prepared_fields(prepared)
        receipt = await publish_trajectory_group(
            coordinator=self.runtime.client, handoff_id=handoff_id,
            members=members, fields=fields, tags=tags, partition_id="train",
            validate_fields=validate_trajectory_fields,
        )
        self._require_finished_publication(receipt)
        return result, receipt

    @staticmethod
    def _require_finished_publication(receipt):
        if (receipt.get("state") != "FINISHED" or receipt.get("prompt_ready_verified") is not True
                or receipt.get("readback_verified") is not True):
            from .trajectory_state.coordinator import PublicationBlocked

            raise PublicationBlocked("trajectory publication did not verify the original prompt's finished status")


def assemble_prepared_fields(prepared):
    """Use stock ragged tensor construction for unequal response lengths."""
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    rows = []
    for batch in prepared:
        if tuple(batch.batch_size) != (1,):
            raise ValueError("each durable outcome must retain exactly one original V1 row")
        row = batch[0]
        rows.append({
            key: value.data if isinstance(value, NonTensorData)
            else value.tolist() if isinstance(value, NonTensorStack) else value
            for key in row.keys() for value in [row.get(key)]
        })
    return list_of_dict_to_tensordict(rows)


def validate_trajectory_fields(fields, *args):
    """Check token alignment; reward/staleness/advantage stay in the V1 Trainer."""
    for row in fields.unbind(0):
        prompt, response, mask = row["prompts"], row["responses"], row["loss_mask"]
        if (prompt.ndim != 1 or response.ndim != 1 or mask.ndim != 1
                or mask.numel() != response.numel() or not torch.all((mask == 0) | (mask == 1))):
            raise ValueError("restored response mask does not align with actual tokens")
        if not torch.equal(row["input_ids"], torch.cat((prompt, response))):
            raise ValueError("restored input tokens differ from original V1 fields")
        logs = row.get("rollout_log_probs")
        if logs is None or logs.numel() != response.numel() or not torch.isfinite(logs).all():
            raise ValueError("restored response lacks actual finite behavior logprobs")
        if not torch.equal(mask, row["response_mask"]):
            raise ValueError("original loss mask changed during handoff")


def state_settings(config):
    settings = config.actor_rollout_ref.rollout.get("trajectory_state", {})
    if not settings.get("enabled", False):
        return None
    from verl.workers.config.rollout import TrajectoryStateConfig

    values = OmegaConf.to_container(settings, resolve=True) if isinstance(settings, DictConfig) else dict(settings)
    values.pop("_target_", None)
    validated = TrajectoryStateConfig(**values)
    rollout = config.actor_rollout_ref.rollout
    agent_name = rollout.agent.default_agent_loop
    agent_config_path = getattr(rollout.agent, "agent_loop_config_path", None)
    if agent_name in {"tool_agent", "durable_tool_agent"}:
        if agent_config_path:
            raise ValueError("the first trajectory_state adapter does not override custom agent loop implementations")
    else:
        # r14 fix (2026-10-16, attempt-9 first-exception-rank-0.trace): the durable
        # business loop is registered through agent_loop_config_path and inherits
        # DurableToolAgentLoop; the exact same inheritance is enforced again at
        # instantiation by _new_loop() (isinstance DurableToolAgentLoop) and the
        # per-sample registry assert. A custom durable loop therefore only needs
        # the explicit registration path; a bare unregistered name stays rejected.
        if not agent_config_path:
            raise ValueError("trajectory_state requires the existing text tool agent or a registered durable loop")
    if not rollout.calculate_log_probs:
        raise ValueError("trajectory handoff requires observed behavior logprobs")
    if type(rollout.n) is not int or rollout.n != 8:
        raise ValueError("the first trajectory_state entry requires rollout.n=8")
    if getattr(rollout, "enable_rollout_routing_replay", False) or rollout.topk_log_probs:
        raise ValueError("trajectory handoff does not support expert replay or top-k heads")
    return validated
