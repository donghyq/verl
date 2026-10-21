"""Checkpoint manager behavior with native Gateway lifecycle hooks."""

from types import MethodType, SimpleNamespace

import pytest

from verl.checkpoint_engine.base import CheckpointEngineManager


@pytest.mark.asyncio
async def test_gateway_stop_happens_before_sleep_abort_and_release():
    events = []

    class Control:
        async def stop_admission(self):
            events.append("stop")

    class Replica:
        async def sleep(self):
            events.append("sleep")

        async def abort_all_requests(self, reject_request=False):
            events.append(("abort", reject_request))

        async def release_kv_cache(self):
            events.append("release")

    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.gateway_control = Control()
    manager.replicas = [Replica()]

    await manager.sleep_replicas()
    await manager.abort_replicas(reject_request=True)
    await manager.release_kv_cache_replicas()

    assert events == ["stop", "sleep", "stop", ("abort", True), "stop", "release"]


@pytest.mark.asyncio
async def test_update_weights_failure_keeps_gateway_stopped():
    events = []

    class Control:
        async def stop_admission(self):
            events.append("stop")

        async def start_admission(self, global_steps):
            events.append(("start", global_steps))

    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.gateway_control = Control()

    async def fail(self, global_steps):
        events.append(("update", global_steps))
        raise RuntimeError("sync failed")

    manager._update_weights = MethodType(fail, manager)

    with pytest.raises(RuntimeError, match="sync failed"):
        await manager.update_weights(9)

    assert events == ["stop", ("update", 9)]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "old_engine", "unsettled_policy", "old_trainer"])
async def test_same_version_copy_needs_actual_readback_before_membership_and_admission(fault):
    events = []
    urls = ("http://hybrid",)

    async def read_versions(addresses):
        assert addresses == urls
        events.append("model-info")
        return {urls[0]: 8 if fault == "old_engine" else 9}

    class Control:
        client = SimpleNamespace(observe_unregistered_engine_versions=read_versions)

        async def stop_admission(self):
            events.append("stop")

        async def start_admission(self, step):
            events.append(("start", step))

        async def prepare_policy_publication(self, step):
            pytest.fail("reborrow must not authorize another weight publication")

    class Publication:
        async def get_active_published_policy(self):
            return {"identity": {"policy_version": "8" if fault == "old_trainer" else "9"}}

        async def authorize_same_version_engines(self, step, addresses, observations):
            assert step == 9 and addresses == list(urls)
            assert observations == [{"engine_url": urls[0], "weight_version": "9", "source": "model_info"}]
            events.append("membership")
            if fault == "unsettled_policy":
                raise ValueError("policy changed while copying")

    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.backend, manager.gateway_control = "naive", Control()
    manager.gateway_control.urls = urls

    async def update(self, step):
        events.append(("copy", step))
        return {}

    manager._update_weights = MethodType(update, manager)
    if fault is None:
        await manager.install_same_version_weights(9, Publication())
        assert events == ["stop", ("copy", 9), "model-info", "membership", ("start", 9)]
    else:
        with pytest.raises(ValueError):
            await manager.install_same_version_weights(9, Publication())
        assert not any(isinstance(item, tuple) and item[0] == "start" for item in events)
        if fault == "old_trainer":
            assert events == []
