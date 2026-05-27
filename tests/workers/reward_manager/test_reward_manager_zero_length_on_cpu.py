# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Regression tests for reward manager index-out-of-bounds when valid_response_length == 0.

When a generation is aborted or truncated to zero tokens, `valid_response_length`
is 0 and `reward_tensor[i, valid_response_length - 1]` silently writes to index
-1 (the last token), corrupting training signal.

Fix: skip the reward assignment when `valid_response_length == 0`.

A secondary bug in `dapo.py`: `self.overlong_buffer_cfg.enable` is accessed
unconditionally, crashing with `AttributeError` when `overlong_buffer_cfg=None`
(the default).

See https://github.com/volcengine/verl/issues/6476
"""

from unittest.mock import MagicMock

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data_proto(batch_size: int, prompt_len: int, resp_len: int, zero_mask_indices=None):
    """Build a minimal DataProto with configurable attention mask.

    Args:
        batch_size: number of samples in the batch.
        prompt_len: number of prompt tokens per sample.
        resp_len: number of response tokens per sample.
        zero_mask_indices: list of batch indices whose response attention mask
            should be all-zero (simulating an empty/aborted generation).
    """
    seq_len = prompt_len + resp_len
    prompts = torch.zeros(batch_size, prompt_len, dtype=torch.long)
    responses = torch.zeros(batch_size, resp_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    if zero_mask_indices:
        for idx in zero_mask_indices:
            attention_mask[idx, prompt_len:] = 0

    td = TensorDict(
        {"prompts": prompts, "responses": responses, "attention_mask": attention_mask},
        batch_size=[batch_size],
    )
    data = DataProto(batch=td)
    data.non_tensor_batch = {
        "data_source": np.array(["test"] * batch_size, dtype=object),
        "reward_model": np.array([{"ground_truth": "ans"}] * batch_size, dtype=object),
    }
    return data


def _dummy_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    return 1.0


# ---------------------------------------------------------------------------
# Tests for naive.py
# ---------------------------------------------------------------------------


class TestNaiveRewardManagerZeroLength:
    """NaiveRewardManager must not write reward to index -1 when valid_response_length==0."""

    def _make_manager(self):
        from verl.workers.reward_manager.naive import NaiveRewardManager

        tokenizer = MagicMock()
        tokenizer.decode = MagicMock(return_value="")
        return NaiveRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=_dummy_compute_score,
        )

    def test_zero_length_response_leaves_tensor_unchanged(self):
        """reward_tensor must stay all-zero when every response has length 0."""
        mgr = self._make_manager()
        data = _make_data_proto(batch_size=2, prompt_len=4, resp_len=8, zero_mask_indices=[0, 1])
        result = mgr(data)
        reward_tensor = result["reward_tensor"] if isinstance(result, dict) else result
        # No reward should have been written anywhere
        assert reward_tensor.sum().item() == 0.0, "Expected all-zero reward tensor for zero-length responses"

    def test_zero_length_does_not_write_to_last_index(self):
        """Index -1 (last position) must remain zero for a zero-length response."""
        mgr = self._make_manager()
        resp_len = 6
        data = _make_data_proto(batch_size=1, prompt_len=3, resp_len=resp_len, zero_mask_indices=[0])
        result = mgr(data)
        reward_tensor = result["reward_tensor"] if isinstance(result, dict) else result
        assert reward_tensor[0, resp_len - 1].item() == 0.0, "Last token position must not be written for zero-length"

    def test_normal_response_still_receives_reward(self):
        """A normal (non-zero) response must still get its reward at the correct position."""
        mgr = self._make_manager()
        prompt_len, resp_len = 3, 5
        data = _make_data_proto(batch_size=1, prompt_len=prompt_len, resp_len=resp_len)
        result = mgr(data)
        reward_tensor = result["reward_tensor"] if isinstance(result, dict) else result
        # Reward should appear at position resp_len - 1 (last valid token)
        assert reward_tensor[0, resp_len - 1].item() != 0.0, "Normal response should receive reward at last valid token"

    def test_mixed_batch_zero_and_normal(self):
        """In a mixed batch, zero-length responses must not corrupt normal ones."""
        mgr = self._make_manager()
        prompt_len, resp_len = 4, 8
        data = _make_data_proto(batch_size=3, prompt_len=prompt_len, resp_len=resp_len, zero_mask_indices=[1])
        result = mgr(data)
        reward_tensor = result["reward_tensor"] if isinstance(result, dict) else result
        # Normal samples (0 and 2) should have reward at last position
        assert reward_tensor[0, resp_len - 1].item() != 0.0
        assert reward_tensor[2, resp_len - 1].item() != 0.0
        # Zero-length sample (1) must not write to index -1
        assert reward_tensor[1, resp_len - 1].item() == 0.0


# ---------------------------------------------------------------------------
# Tests for prime.py
# ---------------------------------------------------------------------------


class TestPrimeRewardManagerZeroLength:
    """PrimeRewardManager must not write reward to index -1 when valid_response_length==0."""

    def _make_manager(self):
        from verl.workers.reward_manager.prime import PrimeRewardManager

        tokenizer = MagicMock()
        tokenizer.batch_decode = MagicMock(return_value=[""] * 10)
        # PrimeRewardManager.verify is called instead of compute_score
        mgr = PrimeRewardManager.__new__(PrimeRewardManager)
        mgr.tokenizer = tokenizer
        mgr.num_examine = 0
        mgr.reward_fn_key = "data_source"
        mgr.verify = MagicMock(return_value=[1.0, 1.0, 1.0])
        return mgr

    def test_zero_length_leaves_tensor_unchanged(self):
        resp_len = 6
        data = _make_data_proto(batch_size=2, prompt_len=3, resp_len=resp_len, zero_mask_indices=[0, 1])
        mgr = self._make_manager()
        mgr.verify = MagicMock(return_value=[1.0, 1.0])
        result = mgr(data)
        reward_tensor = result["reward_tensor"] if isinstance(result, dict) else result
        assert reward_tensor.sum().item() == 0.0

    def test_normal_response_gets_reward(self):
        prompt_len, resp_len = 3, 5
        data = _make_data_proto(batch_size=1, prompt_len=prompt_len, resp_len=resp_len)
        mgr = self._make_manager()
        mgr.verify = MagicMock(return_value=[1.0])
        result = mgr(data)
        reward_tensor = result["reward_tensor"] if isinstance(result, dict) else result
        assert reward_tensor[0, resp_len - 1].item() == 1.0


# ---------------------------------------------------------------------------
# Tests for dapo.py
# ---------------------------------------------------------------------------


class TestDapoRewardManagerZeroLength:
    """DapoRewardManager: zero-length guard and overlong_buffer_cfg=None safety."""

    def _make_manager(self, overlong_buffer_cfg=None):
        from verl.workers.reward_manager.dapo import DapoRewardManager

        tokenizer = MagicMock()
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token = "</s>"
        return DapoRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=_dummy_compute_score,
            overlong_buffer_cfg=overlong_buffer_cfg,
        )

    def test_zero_length_leaves_tensor_unchanged(self):
        """DapoRewardManager must not write to index -1 for zero-length responses."""
        mgr = self._make_manager()
        resp_len = 6
        data = _make_data_proto(batch_size=2, prompt_len=3, resp_len=resp_len, zero_mask_indices=[0, 1])
        result = mgr(data)
        reward_tensor = result["reward_tensor"] if isinstance(result, dict) else result
        assert reward_tensor.sum().item() == 0.0

    def test_overlong_buffer_cfg_none_does_not_crash(self):
        """DapoRewardManager(overlong_buffer_cfg=None) must not raise AttributeError."""
        mgr = self._make_manager(overlong_buffer_cfg=None)
        data = _make_data_proto(batch_size=2, prompt_len=3, resp_len=5)
        # Must not raise
        result = mgr(data)
        reward_tensor = result["reward_tensor"] if isinstance(result, dict) else result
        # Both samples have full responses, so rewards must be assigned
        assert reward_tensor[:, -1].sum().item() != 0.0

    def test_normal_response_gets_reward(self):
        mgr = self._make_manager()
        prompt_len, resp_len = 3, 5
        data = _make_data_proto(batch_size=1, prompt_len=prompt_len, resp_len=resp_len)
        result = mgr(data)
        reward_tensor = result["reward_tensor"] if isinstance(result, dict) else result
        assert reward_tensor[0, resp_len - 1].item() == 1.0

    def test_mixed_batch_zero_and_normal(self):
        mgr = self._make_manager()
        prompt_len, resp_len = 4, 8
        data = _make_data_proto(batch_size=3, prompt_len=prompt_len, resp_len=resp_len, zero_mask_indices=[1])
        result = mgr(data)
        reward_tensor = result["reward_tensor"] if isinstance(result, dict) else result
        assert reward_tensor[0, resp_len - 1].item() != 0.0
        assert reward_tensor[2, resp_len - 1].item() != 0.0
        assert reward_tensor[1, resp_len - 1].item() == 0.0
