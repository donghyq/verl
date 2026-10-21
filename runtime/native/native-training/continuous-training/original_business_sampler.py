"""Sample whole groups through the sealed factory and native reward manager."""
from __future__ import annotations
import asyncio
from copy import deepcopy
import json
import math
from pathlib import Path
import time
from uuid import uuid4


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def require_complete_output(output):
    messages = output.extra_fields.get("messages", {}).get("messages", [])
    if (not output.extra_fields.get("business_reward_messages_complete") or not messages
            or messages[-1].get("role") != "assistant"):
        raise ValueError("Original multi-turn business did not reach its real terminal boundary")
    if len(output.response_ids) != len(output.response_mask) or len(output.response_ids) != len(output.response_logprobs or []):
        raise ValueError("Original response tokens, masks and rollout logprobs are misaligned")


class OriginalBusinessSampler:
    def __init__(self, trainer, *, contract_path, auth_path, rows, policy_record, output, concurrency=8):
        self.trainer, self.contract, self.auth = trainer, str(contract_path), str(auth_path)
        self.rows, self.record, self.output = deepcopy(rows), deepcopy(policy_record), Path(output)
        self.concurrency = concurrency

    def loop(self, server):
        from business_adapter import create_business_loop
        from verl.experimental.agent_loop.agent_loop import DictConfigWrap
        from verl.utils.dataset.rl_dataset import RLHFDataset
        config = self.trainer.config
        return create_business_loop(business_contract_path=self.contract, business_auth_file=self.auth,
            trainer_config=DictConfigWrap(config=config), server_manager=server,
            tokenizer=self.trainer.tokenizer, processor=None, dataset_cls=RLHFDataset,
            data_config=DictConfigWrap(config=config.data), hf_model_type="qwen3_5_moe")

    async def initial_prefixes(self):
        result = []
        for row in self.rows:
            loop = self.loop(None)
            schemas = [loop.tools[name].tool_schema.model_dump(exclude_unset=True, exclude_none=True)
                       for name in loop.business_initial_tools]
            result.append(await loop.ct_build_initial_tokens(deepcopy(row["prompt"]), tools=schemas))
        return result

    async def score(self, output, row, kwargs, directory):
        # Exact construction already exercised by validate_real_service.py:
        # BusinessRewardManager -> original DAPO -> original HTTP scorer.
        import numpy as np
        import torch
        from tensordict import TensorDict
        from verl.protocol import DataProto
        from verl.trainer.ppo.reward import load_reward_manager
        config = deepcopy(self.trainer.config)
        config.trainer.default_local_dir = str(directory / "native-reward-logs")
        manager = load_reward_manager(config, self.trainer.tokenizer,
                                      reward_router_address=None, reward_model_tokenizer=None)
        if manager.__class__.__name__ != "BusinessRewardManager":
            raise ValueError("The actual configured original business reward manager is required")
        def array(value):
            result = np.empty(1, dtype=object); result[0] = deepcopy(value); return result
        non_tensor = {key: array(value) for key, value in kwargs.items()}
        non_tensor.update(data_source=array(row["data_source"]), reward_model=array(row["reward_model"]),
            tool_extra_fields=array(output.extra_fields), __num_turns__=array(output.num_turns),
            prompt_len=array(len(output.prompt_ids)), response_len=array(len(output.response_ids)))
        batch = TensorDict({"prompts": torch.tensor([output.prompt_ids], dtype=torch.int64),
            "responses": torch.tensor([output.response_ids], dtype=torch.int64),
            "attention_mask": torch.ones((1, len(output.prompt_ids) + len(output.response_ids)), dtype=torch.int64)}, batch_size=[1])
        result = await manager.run_single(DataProto(batch=batch, non_tensor_batch=non_tensor))
        score = result.get("reward_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Original native reward did not return a finite score")
        save(directory / "original-reward-result.json", result)
        return score

    async def window(self, client, *, arm, index, version, salt, worker_map, allocated_gpu_count, deadline):
        directory = self.output / f"window-{index}-{arm}"
        directory.mkdir(parents=True, exist_ok=False)
        semaphore = asyncio.Semaphore(self.concurrency)
        rollout = self.trainer.config.actor_rollout_ref.rollout
        params = {"temperature": rollout.temperature, "top_p": rollout.top_p, "top_k": rollout.top_k,
                  "repetition_penalty": 1.0, "logprobs": rollout.calculate_log_probs}
        if rollout.topk_log_probs:
            params["topk_log_probs"] = rollout.topk_log_probs
        started = time.monotonic()
        member_rows = []
        async def group(group_index, row):
            uid = "routing-" + uuid4().hex
            async def member(member_index):
                item = {"uid": uid, "member_index": member_index, "complete": False}
                member_dir = directory / str(group_index) / str(member_index)
                calls = []
                async with semaphore:
                    try:
                        class ObservedGateway:
                            async def generate(self, request_id, **arguments):
                                output = await client.generate(request_id, cache_salt=salt, **arguments)
                                extra = output.extra_fields
                                if (extra.get("engine_weight_version") != str(version) or extra.get("salt_match") is not True
                                        or extra.get("gateway_worker_id") not in worker_map
                                        or not output.token_ids or len(output.log_probs or []) != len(output.token_ids)):
                                    raise ValueError("Actual token, logprob, worker or policy-version proof is missing")
                                call = {"request_id": request_id, "arguments": arguments,
                                    "actual_output": output.model_dump(),
                                    "actual_engine_url": worker_map[extra["gateway_worker_id"]]}
                                calls.append(call); save(member_dir / f"generate-{len(calls)}.json", call)
                                return output
                        loop = self.loop(ObservedGateway())
                        kwargs = deepcopy(row)
                        kwargs.update(raw_prompt=deepcopy(row["prompt"]), uid=uid,
                                      session_id=member_index, global_steps=version)
                        save(member_dir / "input.json", {"kwargs": kwargs, "sampling_params": params})
                        output = await asyncio.wait_for(loop.run(deepcopy(params), **kwargs),
                                                       timeout=max(.01, deadline - time.time()))
                        save(member_dir / "unscored-agent-loop-output.json", output.model_dump())
                        require_complete_output(output)
                        reward = await asyncio.wait_for(self.score(output, row, kwargs, member_dir),
                                                       timeout=max(.01, min(90, deadline - time.time())))
                        output.reward_score = reward
                        save(member_dir / "scored-agent-loop-output.json", output.model_dump())
                        messages = output.extra_fields["messages"]["messages"]
                        item.update(complete=True, reward=reward,
                            returned_tool_message_count=sum(x.get("role") == "tool" for x in messages),
                            declared_tool_names=[c["function"]["name"] for x in messages for c in x.get("tool_calls", [])])
                    except Exception as error:
                        item.update(error_type=type(error).__name__, error=str(error))
                    cached = [x["actual_output"]["extra_fields"].get("num_cached_tokens") for x in calls]
                    item.update(generated_tokens=sum(len(x["actual_output"]["token_ids"]) for x in calls),
                                cached_tokens=sum(cached) if all(type(x) is int for x in cached) else None,
                                model_requests=len(calls))
                    save(member_dir / "receipt.json", item)
                    member_rows.append(item)
                    return item
            members = await asyncio.gather(*(member(i) for i in range(8)))
            return {"uid": uid, "source_group_index": group_index, "members": members,
                    "complete": all(x["complete"] for x in members),
                    "completion_seconds": time.monotonic() - started}
        groups = await asyncio.gather(*(group(i, row) for i, row in enumerate(self.rows)))
        wall = time.monotonic() - started
        result = {"arm": arm, "window_index": index, "status": "COMPLETE" if all(x["complete"] for x in groups) else "INCOMPLETE",
            "policy_version": str(version), "expected_groups": len(self.rows),
            "legal_complete_groups": sum(x["complete"] for x in groups),
            "group_completion_seconds": [x["completion_seconds"] for x in groups if x["complete"]],
            "wall_s": wall, "allocated_gpu_count": allocated_gpu_count,
            "generated_tokens": sum(x.get("generated_tokens", 0) for x in member_rows),
            "cached_tokens_process_metric": sum(x["cached_tokens"] for x in member_rows)
                if all(x.get("cached_tokens") is not None for x in member_rows) else None,
            "groups": groups, "token_equality_prewaived": True, "optimizer_updates": 0}
        save(directory / "window.json", result)
        return result
