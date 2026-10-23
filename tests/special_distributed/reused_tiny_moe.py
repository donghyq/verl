# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# Extracted from test_vllm_keep_requests.py; inference backend imports are excluded.

"""Existing tiny MoE/SGD/NCCL fixture, with no inference-backend import."""

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3MoeConfig

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register


class WeightSender(Worker):
    """Real checkpoint sender with a tiny model and optional next-token SGD updates."""

    def __init__(self, model_path, checkpoint_config):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        self.checkpoint_engine = CheckpointEngineRegistry.new(
            "nccl",
            bucket_size=checkpoint_config.update_weights_bucket_megabytes << 20,
            is_master=True,
            multi_sender=False,
        )
        self.change_weights = False
        self.updates = []

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def enable_updates(self):
        self.change_weights = True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps=None, mode=None):
        if self.change_weights:
            self.optimizer.zero_grad()
            tokens = torch.tensor([[1, 4, 5, 6, 7, 8]], device="cuda")
            loss = self.model(tokens, labels=tokens).loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.updates.append((float(loss.detach()), float(grad_norm)))
        await self.checkpoint_engine.send_weights(self.model.named_parameters(), global_steps=global_steps)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_results(self):
        return self.updates


def _save_tiny_moe(path):
    torch.manual_seed(17)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=2048,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    model.save_pretrained(path)
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0"))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="t0", pad_token="t0", bos_token="t1", eos_token="t2"
    ).save_pretrained(path)
