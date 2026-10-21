from dataclasses import asdict
from types import SimpleNamespace

import pytest

from verl.experimental.fully_async_policy.rollout_artifact import CanonicalPolicyKVIdentity
from verl.experimental.fully_async_policy.trajectory_snapshot import TrajectoryOwnerLease, token_digest
from verl.experimental.fully_async_policy.transfer_admission import TransferAdmissionController, TransferAdmissionConfig
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer


class Remote:
    def __init__(self, fn):
        self.fn = fn
    async def remote(self, *args):
        return self.fn(*args)


def server():
    controller = TransferAdmissionController(TransferAdmissionConfig(shadow=False))
    actor = SimpleNamespace(
        begin_optional_restore=Remote(lambda req: controller.begin(req).to_dict()),
        complete_optional_restore=Remote(controller.complete),
        check_trajectory_owner=Remote(lambda lease: None),
    )
    instance = object.__new__(SGLangHttpServer)
    instance.policy_kv_identity = CanonicalPolicyKVIdentity("0", "weights", "template", "kv")
    instance._disaggregation_role = "null"
    instance.tokenizer_manager = SimpleNamespace(server_args=SimpleNamespace(
        hicache_storage_prefetch_policy="wait_complete"))
    instance.configure_optional_restore_admission(actor, "node-b/eth0")
    calls = []
    async def generate(**kwargs):
        calls.append(kwargs)
        assert controller.summary()["transfer_admission/active"] in (0, 1)
        return TokenOutput(token_ids=[99])
    instance._generate = generate
    return instance, controller, actor, calls


def kwargs(instance):
    tokens = [1, 2, 3, 4]
    return dict(prompt_ids=tokens, sampling_params={}, request_id="request-b",
        allow_remote_cache_restore=True, trajectory_execution_generation=1,
        trajectory_owner_lease=TrajectoryOwnerLease("t", "b", 1, "lease", 99999),
        trajectory_kv_ref={"export_complete": True, "prefix_tokens": 2,
            "token_digest": token_digest(tokens[:2]), "policy_identity": asdict(instance.policy_kv_identity),
            "kv_bytes": 60, "storage_reference": {"backend": "mooncake", "master_address": "store:50051",
                "source_completed_tokens": 2, "objects": [{"key": "page", "bytes": 60, "sha256": "a" * 64}]}})


@pytest.mark.asyncio
async def test_completed_prefix_budget_is_acquired_before_native_generate():
    instance, controller, _, calls = server()
    output = await instance.generate(**kwargs(instance))
    assert output.token_ids == [99]
    assert calls[0]["allow_remote_cache_restore"] is True
    assert calls[0]["remote_cache_restore_prefix_tokens"] == 2
    assert controller.summary()["transfer_admission/max_observed_inflight_bytes"] == 60
    assert controller.summary()["transfer_admission/active"] == 0


@pytest.mark.asyncio
async def test_identity_mismatch_never_starts_optional_restore():
    instance, controller, _, calls = server()
    values = kwargs(instance)
    instance.policy_kv_identity = CanonicalPolicyKVIdentity("1", "new-weights", "template", "kv")
    await instance.generate(**values)
    assert calls[0]["allow_remote_cache_restore"] is False
    assert calls[0]["cache_salt"] == instance.policy_kv_identity.cache_salt
    assert controller.summary()["transfer_admission/decisions"] == 0


@pytest.mark.asyncio
async def test_owner_invalidated_while_queued_does_not_start_native_generate():
    instance, controller, actor, calls = server()
    def stale(_):
        raise ValueError("execution owner is stale")
    actor.check_trajectory_owner = Remote(stale)
    with pytest.raises(RuntimeError, match="owner"):
        await instance.generate(**kwargs(instance))
    assert not calls
    assert controller.summary()["transfer_admission/active"] == 0


def test_execution_mode_requires_confirmable_backend_completion():
    instance, _, actor, _ = server()
    instance.tokenizer_manager.server_args.hicache_storage_prefetch_policy = "timeout"
    with pytest.raises(ValueError, match="wait_complete"):
        instance.configure_optional_restore_admission(actor, "node-b/eth0")
