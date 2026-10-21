import pytest

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    remote_cache_restore_request_gate_enabled,
    resolve_remote_cache_restore_admission,
)


class Config(dict):
    __getattr__ = dict.__getitem__


def config(enforce):
    remote = Config(enforce_request_gate=enforce)
    if enforce:
        remote["cost_gate"] = {
            "restore_model": {
                "fixed_overhead_s": 0.1,
                "bandwidth_bytes_per_s": 1_000_000_000.0,
            },
            "recompute_model": {
                "prefill_tokens_per_second": 1000.0,
            },
            "safety_margin_s": 0.0,
            "min_savings_s": 0.0,
            "bytes_per_token": 1.0,
        }
    return Config(async_training=Config(remote_kv_restore=remote))


def test_remote_restore_request_gate_preserves_existing_behavior_when_disabled():
    assert resolve_remote_cache_restore_admission(config(False), {}) is True
    assert resolve_remote_cache_restore_admission(
        config(False), {"remote_restore_approved": False}
    ) is True
    class Loop(AgentLoopBase):
        async def run(self, sampling_params, **kwargs):
            raise NotImplementedError

    loop = object.__new__(Loop)
    loop.config = config(False)
    assert loop._remote_cache_restore_generate_kwargs({}) == {}


def test_remote_restore_request_gate_fails_closed_when_enforced():
    assert resolve_remote_cache_restore_admission(config(True), {}) is False
    assert resolve_remote_cache_restore_admission(
        config(True), {"remote_restore_approved": False}
    ) is False
    assert resolve_remote_cache_restore_admission(
        config(True), {"remote_restore_approved": True, "remote_prefix_tokens": 0}
    ) is False


def test_remote_restore_request_gate_ignores_external_non_bool_approval():
    assert resolve_remote_cache_restore_admission(
        config(True), {"remote_restore_approved": "true", "remote_prefix_tokens": 10}
    ) is False


def test_remote_restore_request_gate_requires_strict_bool_config():
    with pytest.raises(ValueError, match="enforce_request_gate must be a bool"):
        remote_cache_restore_request_gate_enabled(config("false"))


def test_remote_restore_cost_gate_rejects_short_prefix_and_approves_long_prefix():
    gate_config = config(True)

    assert resolve_remote_cache_restore_admission(
        gate_config, {"remote_prefix_tokens": 10}
    ) is False
    assert resolve_remote_cache_restore_admission(
        gate_config, {"remote_prefix_tokens": 1000}
    ) is True
