# Copyright 2026 Individual Contributor: Hanyudong
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

if sys.platform == "darwin":
    pytest.importorskip("tensordict", reason="The pinned TensorDict runtime is available on Linux")

from verl.workers.config.rollout import SGLangGatewayConfig  # noqa: E402


def test_shared_gpu_gateway_flag_survives_real_hydra_instantiation():
    config = OmegaConf.create(
        {
            "_target_": "verl.workers.config.rollout.SGLangGatewayConfig",
            "enabled": True,
            "url": "http://127.0.0.1:1234",
            "allow_partial_worker_set": True,
        }
    )
    result = instantiate(config)
    assert isinstance(result, SGLangGatewayConfig)
    assert result.allow_partial_worker_set is True
    assert SGLangGatewayConfig().allow_partial_worker_set is False


@pytest.mark.parametrize("invalid", ["true", 1, None])
def test_shared_gpu_gateway_flag_rejects_ambiguous_values(invalid):
    with pytest.raises(ValueError, match="must be boolean"):
        SGLangGatewayConfig(allow_partial_worker_set=invalid)
