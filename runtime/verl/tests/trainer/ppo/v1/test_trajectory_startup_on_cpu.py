"""Startup recovery uses the production entry before ordinary prompt reissue."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
import transfer_queue as tq
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu


def trainer(manager, *, enabled=True):
    return SimpleNamespace(
        config=OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "trajectory_state": {
                            "enabled": enabled,
                            "continuous_policy_publication": True,
                        }
                    }
                }
            }
        ),
        agent_loop_manager=manager,
    )


def test_startup_recovery_requires_completed_publication():
    calls = []
    manager = SimpleNamespace(
        list_recoverable_handoffs=lambda: [{"group_uid": "saved"}],
        recover_trajectory_handoffs=lambda: calls.append("recover") or [{"status": "resumed", "uid": "saved"}],
    )
    value = trainer(manager)
    PPOTrainer._recover_trajectory_handoffs(value)
    assert calls == ["recover"] and value._trajectory_recovered_uids == {"saved"}
    manager.recover_trajectory_handoffs = lambda: [{"status": "unknown", "uid": "saved"}]
    with pytest.raises(RuntimeError, match="did not finish publication"):
        PPOTrainer._recover_trajectory_handoffs(value)
    assert value._trajectory_recovered_uids == set()


def test_disabled_startup_keeps_the_original_path():
    def unexpected():
        raise AssertionError("disabled feature must not query trajectory storage")

    value = trainer(SimpleNamespace(list_recoverable_handoffs=unexpected), enabled=False)
    PPOTrainer._recover_trajectory_handoffs(value)
    assert value._trajectory_recovered_uids == set()


def test_recovered_group_data_is_not_cleared_or_reissued():
    tq.init()
    partition = "trajectory-startup-" + uuid4().hex
    recovered, ordinary = uuid4().hex, uuid4().hex
    trajectory_key = recovered + "_0_0"
    batches = []
    value = SimpleNamespace(
        trainer_mode="separate_async",
        global_steps=8,
        _trajectory_recovered_uids={recovered},
        agent_loop_manager=SimpleNamespace(generate_sequences=batches.append),
    )
    try:
        for uid in (recovered, ordinary):
            tq.kv_batch_put(
                keys=[uid],
                partition_id=partition,
                fields=tu.get_tensordict({"uid": [uid], "raw_prompt": ["original prompt"]}),
                tags=[{"is_prompt": True, "status": "running", "global_steps": 7}],
            )
        tq.kv_put(
            key=trajectory_key,
            partition_id=partition,
            fields={"original": 42},
            tag={"is_prompt": False, "status": "success"},
        )
        assert PPOTrainer._reissue_inflight_prompts(value, partition) == 1
        assert batches[0].get_non_tensor("uid") == [ordinary]
        items = tq.kv_list(partition)[partition]
        assert items[recovered]["status"] == "running" and trajectory_key in items
    finally:
        keys = list(tq.kv_list(partition).get(partition, {}))
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition)
        tq.close()
