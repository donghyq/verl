from types import SimpleNamespace

import pytest
import torch

from verl.experimental.fully_async_policy.rollout_artifact import (
    rollout_training_semantics_from_batch, _encode_training_semantics, _decode_training_semantics,
)


def batch():
    return SimpleNamespace(batch={'responses':torch.tensor([[10,11,99,20,21]]),
        'response_mask':torch.tensor([[1,1,0,1,1]]),
        'rollout_log_probs':torch.tensor([[-0.1,-0.2,0.0,-0.3,-0.4]])},
        non_tensor_batch={'trajectory_segments':[[
            {'start':0,'end':2,'policy_version':'0','sampling_config_fingerprint':'a'*64,
                'turn_index':0,'execution_generation':0},
            {'start':2,'end':4,'policy_version':'0','sampling_config_fingerprint':'a'*64,
                'turn_index':1,'execution_generation':1}]]})


def semantics(data):
    return rollout_training_semantics_from_batch(data, policy_version='0', environment_version='env-1',
        tool_version='tools-1',producer_generation=7)


def test_executor_generation_survives_complete_artifact_serialization():
    result=semantics(batch())
    assert result.producer_generation==7
    assert [s.execution_generation for s in result.segments]==[0,1]
    assert result.generated_token_ids==(10,11,20,21)
    assert _decode_training_semantics(_encode_training_semantics(result))==result
    original=batch()
    for item in original.non_tensor_batch['trajectory_segments'][0]:item.pop('execution_generation')
    legacy=semantics(original)
    encoded=_encode_training_semantics(legacy)
    assert all('execution_generation' not in s for s in encoded['segments'])
    assert _decode_training_semantics(encoded)==legacy


def test_wrong_executor_generation_is_rejected():
    data=batch()
    data.non_tensor_batch['trajectory_segments'][0][1]['execution_generation']=True
    with pytest.raises(ValueError,match='execution generation'):
        semantics(data)
