"""Benchmark adapters over existing native AgentLoop/coordinator/KV codecs.

Imports that load tokenizer/Torch or perform I/O are deferred to explicit CLI
prepare/phase calls. No Trainer, optimizer, new restore or routing algorithm.
"""
from __future__ import annotations
import asyncio
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from recovery_cost_hooks import span, observe_backend, record_native_recovery
from types import SimpleNamespace
from uuid import uuid4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def policy_from_receipt(value):
    if "active_policy_record" in value:
        value = value["active_policy_record"]
        if isinstance(value, str): value = json.loads(value)
    if "identity" in value: return value["identity"]
    if "agent_loop_output" in value:
        return value["agent_loop_output"]["extra_fields"]["trajectory_segments"][0]["policy_identity"]
    if "model_weights_fingerprint" in value: return value
    raise ValueError("A real existing engine/publication/member policy receipt is required")


class AsyncOriginalCoordinator:
    """Only adapt the original synchronous local API to its native async client protocol."""
    def __init__(self, original, cost_log): self.original, self.cost_log = original, cost_log
    def __getattr__(self, key):
        original = getattr(self.original, key)
        async def call(*args, **kwargs):
            stages={"get_handoff_member_record":"semantic_member_record_read",
                    "load_trajectory_snapshot":"semantic_snapshot_read",
                    "record_handoff_member_result":"target_completed_member_persist"}
            if key in stages:
                async with span(self.cost_log, stages[key]):
                    return original(*args, **kwargs)
            return original(*args, **kwargs)
        return call


class Context:
    def __init__(self, spec, root):
        self.spec, self.root = spec, Path(root)
        sys.dont_write_bytecode = True
        sys.path[:0] = [spec["core_source_root"], *spec["business_python_paths"]]
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
        self.config = OmegaConf.load(spec["native_config"])
        self.options = json.loads(Path(spec["business_options"]).read_text())
        self.policy = policy_from_receipt(json.loads(Path(spec["policy_reference"]).read_text()))
        self.sampler_module = load_module("closeout_original_business_sampler", spec["original_sampler"])
        self.http_module = load_module("closeout_original_http_continuation", spec["original_http_continuation"])
        prefix = "verl.trainer.ppo.v1.trajectory_state."
        self.native = SimpleNamespace(**{name: importlib.import_module(prefix + name) for name in
            ("snapshot", "identity", "coordinator", "runtime", "token_kv_resume", "token_kv_source", "preemptible_tool_agent_loop")})
        # Exact R6 standalone transport, using the current core's snapshot and
        # restore contracts through relative imports. No Gateway is invented.
        self.source_codec = load_module(prefix + "closeout_original_standalone_source", spec["original_standalone_source"])
        self.native.identity._decode_policy_identity(self.policy).validate()
        self.model = Path(spec["model_path"])
        frozen = json.loads(Path(spec["native_freeze"]).read_text())
        if (sha(spec["native_freeze"]) != spec["native_freeze_sha256"] or
                sha(spec["native_config"]) != frozen["config_sha256"] or
                self.model.resolve() != Path(frozen["model_path"]).resolve() or
                any(self.policy[k] != frozen["policy_identity"][k] for k in
                    ("model_weights_fingerprint", "tokenizer_template_fingerprint", "kv_cache_schema_fingerprint"))):
            raise ValueError("Current native model/input freeze differs from the original fixed policy")
        # R6's unchanged owner starts initial HF version 0. It must not be used
        # to relabel initial weights as a trained checkpoint/version.
        if self.policy["policy_version"] != "0":
            raise ValueError("R6 benchmark owner loads actual original HF policy 0 only")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model), trust_remote_code=True)
        self.trainer_view = SimpleNamespace(config=self.config, tokenizer=self.tokenizer)
        self.rows = self._rows()
        auth = Path(self.options["business_auth"])
        if not auth.is_file() or auth.stat().st_mode & 0o077:
            raise ValueError("The original business auth file must exist privately; its contents are never recorded")
        self.sampler = self.sampler_module.OriginalBusinessSampler(self.trainer_view,
            contract_path=self.options["business_contract"], auth_path=auth, rows=self.rows,
            policy_record=self.policy, output=root, concurrency=spec["member_concurrency"])
        rollout = self.config.actor_rollout_ref.rollout
        if rollout.n != 8 or rollout.calculate_log_probs is not True:
            raise ValueError("Original n=8 and real rollout logprobs are required")
        self.params = {"temperature": rollout.temperature, "top_p": rollout.top_p, "top_k": rollout.top_k,
                       "repetition_penalty": 1.0, "logprobs": rollout.calculate_log_probs}
        if rollout.topk_log_probs: self.params["topk_log_probs"] = rollout.topk_log_probs
        self.environment = digest({"contract": sha(self.options["business_contract"]), "native_config": sha(spec["native_config"])})
        self.tools = sha(self.options["business_contract"])
        self.events = []
        self.new_tokens = 0
        self.failure_counts = {}

    def _rows(self):
        import pyarrow.parquet as pq
        path = self.options["train_parquet"]
        if sha(path) != self.options["train_sha256"]: raise ValueError("Original Parquet bytes changed")
        rows = pq.read_table(path).to_pylist()
        indices = self.spec.get("source_indices", self.options.get("source_indices", list(range(8))))
        if (len(indices) < 2 or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)):
            raise ValueError("Pin at least two real original records to measure a shared prefix")
        return [rows[i] for i in indices]

    async def fixture(self):
        prompts = await self.sampler.initial_prefixes()
        common = os.path.commonprefix(prompts)
        lengths = [len(x) for x in prompts]
        result = {"initial_prompt_lengths": lengths, "common_prefix_tokens": len(common),
                  "common_prefix_sha256": digest(common), "common_prefix_fractions": [len(common)/x for x in lengths],
                  "fixture_sha256": digest(self.rows), "sampling_sha256": digest(self.params),
                  "policy_identity_sha256": digest(self.policy),
                  "original_reward_implementation_sha256": self.spec["original_reward_implementation_sha256"],
                  "model_config": json.loads((self.model / "config.json").read_text()),
                  "original_enable_prefix_caching": self.config.actor_rollout_ref.rollout.get("enable_prefix_caching"),
                  "actual_cached_counter": None, "gdn_cache_support": None,
                  "cache_support_reason": "Only actual engine snapshot/restore receipts and returned cache counters can establish runtime support",
                  "token_equality_prewaived": True, "original_data_or_generation_budget_modified": False}
        # The agreed 6--7K range describes the full first rendered prompt.
        # Record the genuine shared prefix separately; never pad either one.
        if not common or any(not 6000 <= length <= 7000 for length in lengths):
            result.update(status="NOT_RUN", reason="ACTUAL_ORIGINAL_FULL_PROMPT_OUTSIDE_6_TO_7K_OR_NO_SHARED_PREFIX_NO_PADDING")
        else: result["status"] = "PREPARED_NOT_EXECUTED"
        save(self.root / "fixture.json", result)
        return result, common

    async def identity(self, backend):
        observed = await backend.inspect()
        value = observed["identity"]
        if (value.get("tp_size") != 2 or value.get("actual_model_config_file_sha256") != sha(self.model / "config.json")
                or any(str(value.get(key)) != self.policy["policy_version"] for key in ("weight_version", "actualweight_version"))):
            raise ValueError("Actual engine model/TP/version differs; physical reuse and benchmark generation rejected")
        return observed

    def transport(self, url, backend, identity, check, directory):
        ordinary = self.http_module.NativeHTTPContinuation(engine_url=url, backend=backend, expected_identity=identity,
            source_codec=self.source_codec, validate_output=self.native.token_kv_resume._actual_attempts,
            response_length=self.config.actor_rollout_ref.rollout.response_length,
            max_model_len=self.spec["engine_profile"]["context_length"], deadline=self.deadline,
            cleanup_reserve_s=20, persist=lambda name, row: save(directory/name, row))
        ordinary.check_owner = check
        context = self
        class Observed:
            def __getattr__(self, key): return getattr(ordinary, key)
            async def generate(self, **kwargs):
                started = time.monotonic()
                output = await ordinary.generate(**kwargs)
                context.new_tokens += len(output.token_ids)
                context.events.append({"event": "actual_model_request_returned", "new_model_tokens": len(output.token_ids),
                    "elapsed_s": time.monotonic()-started, "prompt_tokens": len(kwargs["prompt_ids"]),
                    "actual_extra_fields": output.extra_fields})
                return output
        return Observed()

    def runtime(self, directory, owner):
        coordinator = self.native.coordinator.DurableTrajectoryCoordinator(str(directory), self.root.name)
        runtime = self.native.runtime.PromptGroupHandoffRuntime(AsyncOriginalCoordinator(coordinator,
            self.root/("source" if owner.startswith("source-") else "target")/"recovery-cost-segments.jsonl"),
            owner_id=owner, incarnation_id=uuid4().hex, policy_identity=self.policy,
            environment_version=self.environment, tool_version=self.tools,
            lease_seconds=1500, snapshot_ttl_seconds=1800, preemption_deadline_s=180)
        return coordinator, runtime

    def original_grpo_gate(self, outputs, uid):
        """Use the real native GRPO helper and the unchanged learning-signal audit.

        Only the completed AgentLoop's final session is present per member.
        This CPU observation does not claim Trainer/TQ admission or an update.
        """
        import numpy as np
        import torch
        from tensordict import TensorDict
        from verl.protocol import DataProto
        from verl.trainer.ppo.core_algos import AdvantageEstimator
        from verl.trainer.ppo.v1.utils import compute_advantage_for_multi_trajectories
        audit=load_module("closeout_original_training_audit",self.spec["original_training_audit"])
        size=max(len(output.response_ids) for output in outputs)
        responses=torch.zeros((8,size),dtype=torch.int64)
        mask=torch.zeros((8,size),dtype=torch.int64)
        scores=torch.zeros((8,size),dtype=torch.float32)
        for index,output in enumerate(outputs):
            length=len(output.response_ids)
            responses[index,:length]=torch.tensor(output.response_ids,dtype=torch.int64)
            mask[index,:length]=torch.tensor(output.response_mask,dtype=torch.int64)
            effective=mask[index].nonzero()
            if not len(effective):raise ValueError("Actual original member has no generated training tokens")
            scores[index,int(effective[-1])]=output.reward_score
        keys=["%s_%d_0"%(uid,index) for index in range(8)]
        data=DataProto(batch=TensorDict({"responses":responses,"response_mask":mask,
            "token_level_rewards":scores,"rm_scores":scores},batch_size=[8]),
            non_tensor_batch={"uid":np.array([uid]*8,dtype=object)})
        result=compute_advantage_for_multi_trajectories(data,keys,AdvantageEstimator.GRPO,
            num_repeat=8,norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo",True),
            config=self.config.algorithm)
        summary=audit.learning_signal_summary({**{key:value for key,value in result.batch.items()},
            "uid":list(result.non_tensor_batch["uid"])},keys,group_size=8,group_count=1)
        return {**summary,"original_native_helpers_used":True,"TQ_admission_executed":False,
                "optimizer_update_executed":False}

    async def source(self, url):
        backend = self.native.token_kv_resume.PortableSGLangHTTPBackend(url, timeout_s=180)
        observed = await self.identity(backend)
        save(self.root/"source"/"actual-engine-inspect-before.json", observed)
        fixture, prefix = await self.fixture()
        if fixture["status"] != "PREPARED_NOT_EXECUTED": raise ValueError(fixture["reason"])
        async def warm_check(): await self.identity(backend)
        warm = self.transport(url, backend, observed["identity"], warm_check, self.root/"source"/"warm")
        await warm.generate(request_id="warm-"+uuid4().hex, prompt_ids=prefix,
                            sampling_params={"temperature": 0., "max_new_tokens": 1, "logprobs": True})
        summaries = []
        for group_index, row in enumerate(self.rows[:self.spec["groups_per_window"]]):
            uid = "business-benefit-"+uuid4().hex
            directory = self.root/"groups"/str(group_index)
            coordinator, runtime = self.runtime(directory/"durable", "source-"+uuid4().hex)
            initial = {**deepcopy(row), "raw_prompt": deepcopy(row["prompt"]), "uid": uid,
                       "global_steps": 0, "sampling_params": deepcopy(self.params)}
            members = [{**deepcopy(initial), "session_id": index} for index in range(8)]
            started = time.monotonic(); source_snapshots = []
            context = self
            async def preempt(group_uid):
                state = await runtime.request_preemption(group_uid)
                return isinstance(state, dict) and state.get("preemption_requested") is True
            worker = SimpleNamespace(config=self.config, trajectory_entry=SimpleNamespace(runtime=runtime),
                                     request_trajectory_preemption=preempt)
            native_backend = self.source_codec.TokenKVSourceBackend(worker, {
                "engine_url": url, "snapshot_storage_dir": str(directory/"physical-kv"),
                "interrupt_after_tokens": 16, "auto_interrupt": True, "release_full_gpu_pool": False}, timeout_s=180)
            observe_backend(native_backend, self.root/"source/recovery-cost-segments.jsonl", source=True)
            async def run_member(index, original, lease, commit_snapshot, check_owner):
                original_http = context.transport(url, backend, observed["identity"], check_owner, directory/"source-http")
                class AtOriginalToolBoundary:
                    async def generate_owned(self, arguments, *, before_generate, **controls):
                        if not before_generate.completed_tool_batches:
                            return await original_http.generate(**arguments)
                        return await native_backend.generate_owned(arguments, before_generate=before_generate, **controls)
                async def commit(snapshot):
                    async with span(context.root/"source/recovery-cost-segments.jsonl", "semantic_payload_persist"):
                        reference = await commit_snapshot(snapshot)
                    if reference is not None:
                        if not snapshot.completed_tool_batches: raise ValueError("No real original tool result before interruption")
                        frame = snapshot.agent_state["extra_fields"][context.native.token_kv_resume.FRAME]
                        context.new_tokens += len(frame["output_ids"])
                        source_snapshots.append({"member_index": index, "snapshot": json.loads(snapshot.encode()),
                                                 "reference": reference, "retained_partial_tokens": len(frame["output_ids"])})
                    return reference
                loop = context.sampler.loop(original_http)
                kwargs = {k: deepcopy(v) for k,v in original.items() if k != "sampling_params"}
                return await loop.run(deepcopy(original["sampling_params"]), **kwargs,
                    trajectory_id=lease.trajectory_id, rollout_group_id=uid, rollout_sample_index=index, rollout_group_size=8,
                    snapshot_policy_identity=context.policy, snapshot_environment_version=context.environment,
                    snapshot_tool_version=context.tools, execution_generation=lease.generation,
                    _trajectory_check_owner=check_owner, _trajectory_checkpoint=commit,
                    _trajectory_token_kv_source_backend=AtOriginalToolBoundary(), _trajectory_source_lease=lease)
            try:
                # Existing runtime's explicit start_members fixture: one real
                # interrupted member and seven unchanged pending members.
                result = await runtime.run_source(uid, members, run_member, start_members=[0])
                if result.status != "handed_off" or len(source_snapshots) != 1:
                    raise ValueError("Original business did not produce exactly one real post-tool physical checkpoint; no resampling")
                summary = {"uid": uid, "group_index": group_index, "handoff_id": result.handoff_id,
                    "first_dispatch_monotonic": started, "source_snapshots": source_snapshots,
                    "native_status": result.status, "source_generation": result.source_generation}
                save(directory/"source.json", summary); summaries.append(summary)
            finally: coordinator.close()
        save(self.root/"source-groups.json", summaries)
        save(self.root/"source"/"model-counters.json", {"actual_new_model_tokens": self.new_tokens, "events": self.events})

    async def target(self, url, arm):
        backend = self.native.token_kv_resume.PortableSGLangHTTPBackend(url, timeout_s=180)
        observe_backend(backend, self.root/"target/recovery-cost-segments.jsonl", source=False)
        observed = await self.identity(backend)
        save(self.root/"target"/"actual-engine-inspect-before.json", observed)
        death = json.loads((self.root/"source-exit.json").read_text())
        integration = load_module("closeout_original_recovery_integration", self.spec["original_recovery_integration"])
        semaphore = asyncio.Semaphore(self.spec["member_concurrency"])
        groups = []
        source_groups = json.loads((self.root/"source-groups.json").read_text())
        for source in source_groups:
            directory = self.root/"groups"/str(source["group_index"])
            coordinator, runtime = self.runtime(directory/"durable", "target-"+uuid4().hex)
            receipts = {}
            async def execute(index, original, lease, check_owner, snapshot=None):
                async with semaphore:
                    member_dir = directory/"target"/str(index)
                    ordinary = self.transport(url, backend, observed["identity"], check_owner, member_dir)
                    loop = self.sampler.loop(ordinary)
                    kwargs = {k: deepcopy(v) for k,v in original.items() if k != "sampling_params"}
                    if snapshot is not None:
                        integration.validate_source_exit_binding(death, snapshot.kv_ref["source_binding"]["source_engine_instance_id"])
                        if (observed["identity"] != snapshot.kv_ref["identity"] or
                                observed["engine_instance_id"] == snapshot.kv_ref["source_binding"]["source_engine_instance_id"]):
                            raise ValueError("Fresh target is not physically compatible with the exact original policy/cache identity")
                        guarded = self.http_module.ReadOnlyToolDispatch(loop.business_manager.call_tool,
                            snapshot.completed_tool_batches, [], lambda value:save(member_dir/"tool-dispatches.json", value))
                        loop.business_manager.call_tool = guarded
                        for tool in loop.tools.values(): tool._manager_call_tool = guarded
                        async def bridge_check(): await check_owner()
                        # Explicit same-policy benchmark choice. Native owner
                        # checks and resume_from_snapshot still use the original
                        # check_owner; no new policy or fake incompatible engine.
                        bridge_check.token_only_resume = arm == "token_recompute"
                        bridge_check.target_policy_identity = self.policy
                        self.native.token_kv_resume.install_token_kv_resume(loop, snapshot=snapshot, lease=lease,
                                                                           check_owner=bridge_check, backend=backend)
                        wrapped = loop.server_manager
                        context = self
                        class ObserveNativeFirst:
                            def __getattr__(self, key): return getattr(wrapped, key)
                            async def generate(self, *args, **kw):
                                output = await wrapped.generate(*args, **kw)
                                recovery = output.extra_fields.get("trajectory_token_kv_recovery", {})
                                if recovery: record_native_recovery(context.root/"target/recovery-cost-segments.jsonl", recovery)
                                if recovery.get("mode") == "PORTABLE_STATE_RESTORED":
                                    frame = snapshot.agent_state["extra_fields"][context.native.token_kv_resume.FRAME]
                                    context.new_tokens += len(output.token_ids)-len(frame["output_ids"])
                                expected = "NATIVE_TOKEN_RECOMPUTE" if arm == "token_recompute" else "PORTABLE_STATE_RESTORED"
                                if recovery and recovery.get("mode") != expected: raise ValueError("Actual original recovery branch differs from frozen arm")
                                if recovery: save(member_dir/"native-recovery.json", {"actual": recovery,
                                    "benchmark_choice": arm, "same_actual_policy": True,
                                    "native_reason_field_is_not_used_as_a_policy_change_proof": True})
                                return output
                        loop.server_manager = ObserveNativeFirst()
                        async with span(context.root/"target/recovery-cost-segments.jsonl", "retained_history_resume_total", arm=arm):
                            output = await loop.resume_from_snapshot(snapshot, check_owner=check_owner,
                                execution_generation=lease.generation, environment_version=self.environment,
                                tool_version=self.tools, **kwargs)
                        state = snapshot.agent_state
                        count = len(state["response_mask"])
                        historical = (output.response_ids[:count] == state["prompt_ids"][-count:]
                            and output.response_mask[:count] == state["response_mask"]
                            and output.response_logprobs[:count] == state["response_logprobs"]
                            and output.extra_fields["messages"]["messages"][:len(state["messages"])] == state["messages"])
                    else:
                        output = await loop.run(deepcopy(original["sampling_params"]), **kwargs,
                            trajectory_id=lease.trajectory_id, rollout_group_id=source["uid"], rollout_sample_index=index,
                            rollout_group_size=8, snapshot_policy_identity=self.policy,
                            snapshot_environment_version=self.environment, snapshot_tool_version=self.tools,
                            execution_generation=lease.generation, _trajectory_check_owner=check_owner)
                        historical = True  # Pending input has no committed generated history.
                    self.sampler_module.require_complete_output(output)
                    if not historical: raise ValueError("Actual native continuation changed committed tokens, logprobs or tool messages")
                    reward = await self.sampler.score(output, self.rows[source["group_index"]], original, member_dir)
                    output.reward_score = reward
                    messages = output.extra_fields["messages"]["messages"]
                    receipts[index] = {"uid": source["uid"], "member_index": index, "complete": True,
                        "history_preserved": historical, "reward": reward,
                        "returned_tool_message_count": sum(x.get("role")=="tool" for x in messages),
                        "effective_training_tokens": sum(output.response_mask), "response_tokens": len(output.response_ids)}
                    save(member_dir/"original-scored-output.json", output.model_dump())
                    return output
            async def resume(index, snapshot, original, lease, check): return await execute(index, original, lease, check, snapshot)
            async def pending(index, original, lease, check): return await execute(index, original, lease, check)
            try:
                result = await runtime.run_target(source["handoff_id"], resume, pending)
                if result.status != "completed" or len(result.outputs) != 8: raise ValueError("Original native group did not complete")
                group = {"uid": source["uid"], "native_runtime_status": result.status,
                         "native_output_record_indices": sorted(result.output_record_refs),
                         "members": [receipts[i] for i in range(8)],
                         "original_grpo_gate":self.original_grpo_gate(result.outputs,source["uid"]),
                         "completion_seconds": time.monotonic()-source["first_dispatch_monotonic"]}
                if not any(x["returned_tool_message_count"] > 0 for x in group["members"]):
                    raise ValueError("The actual original group did not contain multi-turn tool work")
                save(directory/"complete-group.json", group); groups.append(group)
            finally: coordinator.close()
        save(self.root/"target-groups.json", groups)
        save(self.root/"target"/"model-counters.json", {"actual_new_model_tokens": self.new_tokens, "events": self.events})


def phase(spec, root, name, url, arm):
    root = Path(root)
    try:
        context = Context(spec, root)
        context.deadline = json.loads((root/"hardware-before.json").read_text())["absolute_deadline"]
        async def run():
            await asyncio.wait_for(getattr(context, name)(url) if name == "source" else context.target(url, arm),
                                   timeout=max(.01, context.deadline-time.time()-25))
        asyncio.run(run())
        save(root/name/"phase-complete.json", {"success": True, "optimizer_updates": 0})
    except BaseException as error:
        save(root/name/"phase-complete.json", {"success": False, "error_type": type(error).__name__, "error": str(error)})
        raise
