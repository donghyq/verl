# Copyright 2026 Individual Contributor: donghyq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in tiny SGLang GPU probe. All observations wrap native methods; no simulated counters."""

import asyncio
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import ray
import torch
from reused_tiny_moe import WeightSender, _save_tiny_moe
from sglang.srt.entrypoints.engine import Engine
from sglang_preserve_history import classify_native_history
from sglang_preserve_process import wait_owned_children

from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import split_resource_pool
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer, SGLangReplica

# These real router parameters are replicated under the native tiny-MoE TP
# layout. Comparing them does not pretend that HF and fused/sharded native
# expert parameter names or storage layouts are interchangeable.
SELECTED = tuple(f"model.layers.{layer}.mlp.gate.weight" for layer in range(2))


def digest(tensor):
    tensor = tensor.detach().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def parameter_digests(model):
    values = {name: digest(value) for name, value in model.named_parameters() if name in SELECTED}
    if set(values) != set(SELECTED):
        raise RuntimeError("Tiny MoE parameter identity does not match this probe")
    return values


def tree_digest(value):
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": digest(value)}
    if isinstance(value, dict):
        return {key: tree_digest(part) for key, part in value.items()}
    if isinstance(value, list | tuple):
        return [tree_digest(part) for part in value]
    raise TypeError("Unknown native KV-copy payload; no counter or digest is inferred")


class RecordedSender(WeightSender):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.send_records = []
        send = self.checkpoint_engine.send_weights

        async def observed_send(weights, **send_kwargs):
            expected = [name for name, _ in self.model.named_parameters()]
            visited, sizes = [], []

            def stream():
                for name, tensor in weights:
                    visited.append(name)
                    sizes.append(tensor.numel() * tensor.element_size())
                    yield name, tensor

            result = await send(stream(), **send_kwargs)  # Original NCCL sender, not a replacement transport.
            assert visited == expected, "Complete trainer parameter stream was not consumed"
            self.send_records.append(
                {"global_steps": send_kwargs.get("global_steps"), "names": visited, "bytes": sum(sizes)}
            )
            return result

        self.checkpoint_engine.send_weights = observed_send

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def weight_evidence(self):
        return {"selected": parameter_digests(self.model), "sends": list(self.send_records), "SGD": self.updates}


def observed_scheduler_process(*args, **kwargs):
    """Run the original scheduler with read-only test observations in its own child process."""
    import sglang.srt.managers.scheduler as native
    from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

    original_scheduler = native.Scheduler

    class ObservedScheduler(original_scheduler):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.probe_completed, self.probe_results, self.probe_finished = [], {}, {}
            self.probe_forward_id = 0
            self.probe_tensor_updates = []
            self.probe_context = None
            runner = self.tp_worker.model_runner
            forward = runner.forward
            update = self.tp_worker.update_weights_from_tensor

            def observe_update(_worker, request):
                result = update(request)  # Native rank-specific IPC deserialization and loader.
                self.probe_tensor_updates.append(
                    {
                        "success": result[0] is True,
                        "flush_cache": request.flush_cache,
                        "abort_all_requests": request.abort_all_requests,
                        "serialized_tp_entries": len(request.serialized_named_tensors),
                    }
                )
                return result

            self.tp_worker.update_weights_from_tensor = types.MethodType(observe_update, self.tp_worker)

            def observe_forward(_runner, forward_batch, *a, **kw):
                if self.probe_context is not None:
                    rows = forward_batch.req_pool_indices.detach().cpu().tolist()
                    positions = forward_batch.positions.detach().cpu().tolist()
                    lengths = (
                        [1] * len(rows)
                        if forward_batch.forward_mode.is_decode()
                        else list(forward_batch.extend_seq_lens_cpu)
                    )
                    if len(rows) != len(lengths) or len(positions) != sum(lengths):
                        raise RuntimeError("Unrecognized actual ForwardBatch position layout")
                    cursor = 0
                    weights = parameter_digests(_runner.model)
                    for row, length in zip(rows, lengths, strict=True):
                        rid = self.probe_context["rows"][int(row)]
                        self.probe_context["inputs"].append(
                            {
                                "rid": rid,
                                "forward_id": self.probe_context["forward_id"],
                                "positions": positions[cursor : cursor + length],
                                "weights": weights,
                            }
                        )
                        cursor += length
                return forward(forward_batch, *a, **kw)

            runner.forward = types.MethodType(observe_forward, runner)

        def run_batch(self, batch, *a, **kw):
            self.probe_forward_id += 1
            context = {
                "forward_id": self.probe_forward_id,
                "rows": {int(req.kv.req_pool_idx): req.rid for req in batch.reqs},
                "inputs": [],
            }
            self.probe_context = context
            try:
                result = super().run_batch(batch, *a, **kw)
            finally:
                self.probe_context = None
            if id(result) in self.probe_results:
                raise RuntimeError("Native result reused before successful processing")
            self.probe_results[id(result)] = context["inputs"]
            return result

        def process_batch_result(self, batch, result):
            requests = list(batch.reqs)
            starts = {req.rid: len(req.output_ids) for req in requests}
            super().process_batch_result(batch, result)
            # Original output processing synchronizes copy_done; this explicit
            # test synchronization additionally rejects asynchronous CUDA errors.
            torch.cuda.synchronize()
            records = self.probe_results.pop(id(result))
            for row in records:
                req = next(req for req in requests if req.rid == row["rid"])
                row["output_interval"] = [starts[req.rid], len(req.output_ids)]
            self.probe_completed.extend(records)
            for req in requests:
                if req.finished():
                    self.probe_finished[req.rid] = self.request_state(req, copy_kv=False)

        def request_state(self, req, copy_kv):
            state = {
                "rid": req.rid,
                "token_ids": list(req.output_ids),
                "log_probs": list(req.logprob.output_token_logprobs_val or []),
                "cache_salt": req.cache_salt,
                "cached_tokens": int(req.cached_tokens),
                "finished": req.finished(),
                "request_row": req.kv.req_pool_idx,
                "kv_committed_len": int(req.kv.kv_committed_len),
                "routing": None,
            }
            routing = req.routed_experts
            if not req.finished():
                capturer = get_global_experts_capturer()
                if capturer is not None and req.return_routed_experts:
                    routing = capturer.get_topk(
                        req_pool_idx=req.kv.req_pool_idx,
                        seqlen=len(req.origin_input_ids) + len(req.output_ids_through_stop),
                        req_to_token_pool=self.req_to_token_pool,
                        start_len=req.routed_experts_start_len,
                    )
            if routing is not None:
                state["routing"] = routing.reshape(-1, 2, 2).cpu().tolist()
            if copy_kv and not req.finished():
                row, length = int(req.kv.req_pool_idx), int(req.kv.kv_committed_len)
                indices = self.req_to_token_pool.req_to_token[row, :length].clone()
                pool = self.token_to_kv_pool_allocator.get_kvcache()
                if not hasattr(pool, "k_buffer") or not hasattr(pool, "v_buffer"):
                    raise RuntimeError("This tiny-attention probe does not recognize the native pool")
                state["token_slots"] = indices.cpu().tolist()
                state["KV"] = tree_digest(pool.get_cpu_copy(indices))
                state["KV_storage"] = [tensor.data_ptr() for tensor in [*pool.k_buffer, *pool.v_buffer]]
            return state

        def get_internal_state(self, request):
            response = super().get_internal_state(request)
            unique = {}
            batches = [self.running_batch, self.last_batch]
            for batch in batches:
                if batch is not None:
                    unique.update({req.rid: req for req in batch.reqs if not req.finished()})
            if self.chunked_req is not None and not self.chunked_req.finished():
                unique[self.chunked_req.rid] = self.chunked_req
            observation = {
                "tp_rank": self.tp_group.rank_in_group,
                "tp_size": self.tp_group.world_size,
                "overlap_enabled": bool(self.enable_overlap),
                "radix_cache_disabled": response.internal_state.get("disable_radix_cache"),
                "last_launched_forward_id": self.probe_forward_id,
                "paused": bool(self._engine_paused),
                "requests": {rid: self.request_state(req, copy_kv=self._engine_paused) for rid, req in unique.items()},
                "completed": list(self.probe_completed),
                "pending": [row for rows in self.probe_results.values() for row in rows],
                "finished": dict(self.probe_finished),
                "weights": parameter_digests(self.tp_worker.model_runner.model),
                "tensor_updates": list(self.probe_tensor_updates),
            }
            # This native control request is broadcast to the TP workers. Gather
            # observations on its existing CPU group; TP0's real response carries
            # the original per-rank evidence instead of fabricating other ranks.
            observations = [None] * self.tp_group.world_size
            torch.distributed.all_gather_object(observations, observation, group=self.tp_cpu_group)
            response.internal_state["preserve_probe"] = observations
            return response

    native.Scheduler = ObservedScheduler
    native.run_scheduler_process(*args, **kwargs)


class ObservedServer(SGLangHttpServer):
    async def launch_server(self, *args, **kwargs):
        descriptor = Engine.__dict__["_launch_subprocesses"]
        launch = Engine._launch_subprocesses

        def observe_launch(**launch_kwargs):
            launch_kwargs["run_scheduler_process_func"] = observed_scheduler_process
            output = launch(**launch_kwargs)
            if len(output) != 6:
                raise RuntimeError("Native launch return contract changed")
            output[0]._subprocess_watchdog = output[4]
            self._weight_cache_daemon_procs = output[5]
            return output

        Engine._launch_subprocesses = staticmethod(observe_launch)
        try:
            await super().launch_server(*args, **kwargs)
        finally:
            Engine._launch_subprocesses = descriptor

    async def observed_native(self):
        states = await self.tokenizer_manager.get_internal_state()
        if len(states) != 1:
            raise RuntimeError("This probe requires native DP1")
        return states[0]["preserve_probe"]

    async def cleanup_owned_engine(self):
        Engine.shutdown(self)  # Native cleanup affects only this owned Ray actor's children.
        return wait_owned_children(timeout=30)


class ObservedReplica(SGLangReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(ObservedServer)
        self.observe, self.boundaries = False, []

    async def pause_generation(self):
        await super().pause_generation()
        if self.observe:
            self.before = await self.servers[0].observed_native.remote()

    async def resume_generation(self):
        if self.observe:
            self.boundaries.append((self.before, await self.servers[0].observed_native.remote()))
        await super().resume_generation()


async def exercise(model_path, out, tp):
    checkpoint = CheckpointEngineConfig(backend="nccl", update_weights_bucket_megabytes=1)
    config = RolloutConfig(
        name="sglang",
        mode="async",
        tensor_model_parallel_size=tp,
        prompt_length=64,
        response_length=512,
        max_model_len=1024,
        max_num_seqs=4,
        max_num_batched_tokens=1024,
        enforce_eager=True,
        preserve_kv_cache=True,
        enable_prefix_caching=True,
        prefix_cache_mode="versioned",
        enable_rollout_routing_replay=True,
        calculate_log_probs=True,
        free_cache_engine=False,
        gpu_memory_utilization=0.2,
        checkpoint_engine=checkpoint,
        engine_kwargs={
            "sglang": {"attention_backend": "triton", "moe_runner_backend": "triton", "max_total_tokens": 2048}
        },
    )
    pool = RayResourcePool(process_on_nodes=[1 + tp], max_colocate_count=2)
    sender_pool, rollout_pool = split_resource_pool(pool, [1, tp])
    sender = RayWorkerGroup(
        resource_pool=sender_pool,
        ray_cls_with_init=RayClassWithInitArgs(
            cls=ray.remote(RecordedSender), model_path=model_path, checkpoint_config=checkpoint
        ),
    )
    replica = ObservedReplica(0, config, HFModelConfig(path=model_path), gpus_per_node=1 + tp)
    cleanup = None
    try:
        await replica.init_standalone(rollout_pool)
        manager = CheckpointEngineManager(checkpoint, sender, [replica])
        await manager.update_weights(global_steps=0)
        before_sender = sender.weight_evidence()[0]
        prompt = [1, 4, 5, 6] * 16
        sampling = {"temperature": 0.0, "max_tokens": 512, "ignore_eos": True, "logprobs": True}
        rid = "crossweight-live"
        pending = replica.servers[0].generate.remote(prompt, dict(sampling), rid)
        async with asyncio.timeout(90):
            while True:
                states = await replica.servers[0].observed_native.remote()
                if sorted(state["tp_rank"] for state in states) != list(range(tp)):
                    raise RuntimeError("Actual native TP observations are incomplete")
                active = states[0]["requests"].get(rid)
                if active and 8 <= len(active["token_ids"]) < 480:
                    break
                if rid in states[0]["finished"]:
                    raise RuntimeError("Request completed before the weight-update window; no retry")
                await asyncio.sleep(0.01)
        sender.enable_updates()
        replica.observe = True
        await manager.update_weights(global_steps=1)
        output = await asyncio.wait_for(pending, timeout=120)
        replica.observe = False
        final_native = await replica.servers[0].observed_native.remote()
        after_sender = sender.weight_evidence()[0]
        fresh = await replica.servers[0].generate.remote(prompt, {**sampling, "max_tokens": 1}, "fresh-version-one")
        next_native = await replica.servers[0].observed_native.remote()
        if len(replica.boundaries) != 1:
            raise RuntimeError("Exactly one observed real update is required")
        before, after = replica.boundaries[0]
        # Keep the actual observations when a strict validation below rejects.
        (out / "failure-diagnostics.json").write_text(
            json.dumps(
                {
                    "validation_status": "NOT_YET_VALIDATED",
                    "prompt_length": len(prompt),
                    "response_length": len(output.token_ids),
                    "public_output": {
                        "token_ids": output.token_ids,
                        "log_probs": output.log_probs,
                        "stop_reason": output.stop_reason,
                        "routed_experts": (None if output.routed_experts is None else output.routed_experts.tolist()),
                        "extra_fields": output.extra_fields,
                    },
                    "public_fresh_extra_fields": fresh.extra_fields,
                    "before": before,
                    "after": after,
                    "final": final_native,
                    "next_native": next_native,
                    "sender_before": before_sender,
                    "sender_after": after_sender,
                },
                indent=2,
            )
            + "\n"
        )
        rank_receipts = [
            validate(b, a, f, before_sender, after_sender, output, fresh, n, rid)
            for b, a, f, n in zip(before, after, final_native, next_native, strict=True)
        ]
        receipt = {
            "status": "PASS_TINY_MOE_CROSSWEIGHT_PRESERVATION",
            "rollout_tp": tp,
            "required_gpus": 1 + tp,
            "ranks": rank_receipts,
        }
        (out / "evidence.json").write_text(
            json.dumps(
                {
                    "receipt": receipt,
                    "before": before,
                    "after": after,
                    "final": final_native,
                    "sender_before": before_sender,
                    "sender_after": after_sender,
                },
                indent=2,
            )
            + "\n"
        )
        return receipt
    finally:
        original_failure = sys.exc_info()[0] is not None
        try:
            if replica.servers:
                cleanup = await replica.servers[0].cleanup_owned_engine.remote()
        except BaseException as exc:
            cleanup = {"status": "FAILED", "cause": type(exc).__name__}
            if not original_failure:
                raise
        finally:
            (out / "cleanup.json").write_text(json.dumps({"owned_engine": cleanup}) + "\n")


def validate(before, after, final, sender_before, sender_after, output, fresh, next_native, rid):
    assert before["tp_rank"] == after["tp_rank"] == final["tp_rank"] == next_native["tp_rank"]
    assert before["paused"] and after["paused"]
    assert before["overlap_enabled"] and after["overlap_enabled"]
    assert before["radix_cache_disabled"] is False and after["radix_cache_disabled"] is False
    assert rid in before["requests"] and rid in after["requests"]
    state = before["requests"][rid]
    assert state["kv_committed_len"] > 0 and len(state["token_slots"]) == state["kv_committed_len"]
    assert state == after["requests"][rid], "Request/KV/routing changed during the update"
    assert before["completed"] == after["completed"], "Forward completed during paused update"
    assert before["pending"] == after["pending"], "Native result ownership changed during paused update"
    assert before["weights"] == sender_before["selected"]
    assert after["weights"] == sender_after["selected"] and before["weights"] != after["weights"]
    assert len(after["tensor_updates"]) > len(before["tensor_updates"]) > 0
    assert all(
        row["success"]
        and row["flush_cache"] is False
        and row["abort_all_requests"] is False
        and row["serialized_tp_entries"] == before["tp_size"]
        for row in after["tensor_updates"]
    )
    assert len(sender_after["SGD"]) == 1
    loss, grad = sender_after["SGD"][0]
    assert np.isfinite(loss) and np.isfinite(grad) and grad > 0
    assert len(sender_after["sends"]) == 2 and all(send["bytes"] > 0 for send in sender_after["sends"])
    count = len(state["token_ids"])
    assert not state["finished"] and 0 < count < len(output.token_ids) == 512
    assert output.stop_reason == "length" and len(output.log_probs) == 512
    assert output.token_ids[:count] == state["token_ids"]
    assert output.log_probs[:count] == state["log_probs"]
    if state["routing"] is None or output.routed_experts is None:
        raise RuntimeError("UNKNOWN: native MoE routing capture is missing")
    np.testing.assert_array_equal(output.routed_experts[: len(state["routing"])], state["routing"])
    assert output.extra_fields["min_global_steps"] == 0 and output.extra_fields["max_global_steps"] == 1
    assert output.extra_fields["cache_salt"] == "0" and fresh.extra_fields["cache_salt"] == "1"
    assert next_native["finished"]["fresh-version-one"]["cached_tokens"] == 0
    assert next_native["finished"]["fresh-version-one"]["cache_salt"] == "1"
    # In overlap mode an old forward can finish before pause but remain queued
    # for native result processing. Its later commit is not a recomputation.
    old = [pos for row in before["completed"] + before["pending"] if row["rid"] == rid for pos in row["positions"]]
    all_completed = [row for row in final["completed"] if row["rid"] == rid]
    # Native overlap may finish one extra forward after the request is done.
    # Keep its compute in the receipt, but do not invent an output for it.
    completed, discarded = classify_native_history(
        all_completed, 64, len(output.token_ids), final["overlap_enabled"], final["finished"][rid]["finished"]
    )
    positions = [pos for row in all_completed for pos in row["positions"]]
    if not old or not positions:
        raise RuntimeError("UNKNOWN: actual completed forward input positions missing")
    new = [row for row in all_completed if row["forward_id"] > before["last_launched_forward_id"]]
    history_overlap = sum(pos in set(old) for row in new for pos in row["positions"])
    assert history_overlap == 0
    assert any(row["weights"] == after["weights"] for row in new), "No completed forward using updated parameters"
    output_intervals = [row["output_interval"] for row in completed]
    assert output_intervals == [[index, index + 1] for index in range(512)]
    assert all(row["weights"] in (before["weights"], after["weights"]) for row in all_completed)
    updated_outputs = sum(
        row["output_interval"][1] - row["output_interval"][0] for row in completed if row["weights"] == after["weights"]
    )
    assert updated_outputs > 0
    return {
        "status": "PASS_TINY_MOE_CROSSWEIGHT_PRESERVATION",
        "RID": rid,
        "tp_rank": before["tp_rank"],
        "optimizer_steps": 1,
        "preserved_output_tokens": count,
        "output_tokens_after_preserved_prefix": 512 - count,
        "outputs_using_updated_router_parameters": updated_outputs,
        "completed_forward_inputs": len(positions),
        "productive_forward_inputs": sum(len(row["positions"]) for row in completed),
        "discarded_terminal_forwards": len(discarded),
        "discarded_terminal_forward_inputs": sum(len(row["positions"]) for row in discarded),
        "historical_inputs_recomputed": history_overlap,
        "completed_native_tensor_buckets": len(after["tensor_updates"]) - len(before["tensor_updates"]),
        "routing_prefix_verified": True,
        "fresh_version_salt": "1",
        "native_weight_version_spans": "UNKNOWN_NOT_ASSERTED",
        "scope": (
            "One tiny MoE, native TP workers+one trainer GPU, in-place synchronization; "
            "no throughput or cross-host claim"
        ),
    }


def run_gpu(out, tp=2):
    if tp not in (1, 2):
        raise ValueError("Only TP1 debug and TP2 acceptance are defined")
    if ray.is_initialized():
        raise RuntimeError("Use a fresh process; this probe must not share an existing Ray session")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    owned_ray = False
    try:
        if torch.cuda.device_count() != 1 + tp:
            raise RuntimeError(f"Requires exactly {1 + tp} dedicated visible GPUs: sender1 + native rollout TP{tp}")
        torch.cuda.init()
        devices = []
        for index in range(1 + tp):
            properties = torch.cuda.get_device_properties(index)
            if properties.major < 8:
                raise RuntimeError("This BF16 tiny-MoE candidate requires SM80 or newer")
            devices.append({"index": index, "name": properties.name, "uuid": str(properties.uuid)})
        (out / "cuda-preflight.json").write_text(
            json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "devices": devices}) + "\n"
        )
        model = out / "tiny_moe"
        _save_tiny_moe(model)
        pythonpath = os.pathsep.join((os.getcwd(), str(Path(__file__).resolve().parent)))
        ray.init(address="local", num_gpus=1 + tp, num_cpus=12, runtime_env={"env_vars": {"PYTHONPATH": pythonpath}})
        owned_ray = True
        receipt = asyncio.run(exercise(str(model), out, tp))
        (out / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return receipt
    except BaseException as exc:
        import traceback

        (out / "first-error.txt").write_text(traceback.format_exc())
        (out / "result.json").write_text(
            json.dumps({"status": "FAILED_OR_INCOMPLETE", "cause": type(exc).__name__}) + "\n"
        )
        raise
    finally:
        if owned_ray:
            ray.shutdown()  # Only the local Ray instance started by this probe.
