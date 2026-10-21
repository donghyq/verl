import asyncio

import pytest

from verl.experimental.fully_async_policy.optional_restore import OptionalRestoreRunner, SettledRestoreFailure
from verl.experimental.fully_async_policy.transfer_admission import (
    TransferAdmissionConfig, TransferAdmissionController, TransferIntent, TransferRequest,
)


def setup_runner(**config):
    controller = TransferAdmissionController(TransferAdmissionConfig(shadow=False,
        max_inflight_transfers=1, max_inflight_bytes=100, **config))
    async def begin(request):
        return controller.begin(request).to_dict()
    async def complete(transfer_id):
        controller.complete(transfer_id)
    return controller, OptionalRestoreRunner(begin, complete, resource_domain="nic-a",
        queue_timeout_seconds=0.05, poll_seconds=0.005)


def request(name, approved=True):
    return TransferRequest(name, TransferIntent.REMOTE_KV_RESTORE, 60,
        cost_gate_approved=approved, metadata={"resource_domain": "nic-a"})


@pytest.mark.asyncio
async def test_admit_precedes_io_and_queue_cannot_start_io_until_completion():
    controller, runner = setup_runner()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def first():
        assert controller.summary()["transfer_admission/active"] == 1
        calls.append("first")
        entered.set()
        await release.wait()
        return 1
    async def second():
        calls.append("second")
        return 2
    a = asyncio.create_task(runner.run(request("a"), first))
    await entered.wait()
    b = asyncio.create_task(runner.run(request("b"), second))
    await asyncio.sleep(0.012)
    assert calls == ["first"]
    release.set()
    assert (await a).value == 1 and (await b).value == 2
    assert controller.summary()["transfer_admission/active"] == 0
    assert controller.summary()["transfer_admission/max_observed_inflight_transfers"] == 1


@pytest.mark.asyncio
async def test_reject_and_bounded_queue_timeout_skip_optional_io():
    controller, runner = setup_runner()
    async def forbidden():
        pytest.fail("rejected/queued KV I/O started")
    assert not (await runner.run(request("deny", False), forbidden)).used_optional_io
    controller.begin(request("busy"))
    result = await runner.run(request("queued"), forbidden)
    assert not result.used_optional_io and result.reason == "queue_timeout"
    assert controller.summary()["transfer_admission/active"] == 1


@pytest.mark.asyncio
async def test_caller_exit_does_not_release_bytes_while_io_is_alive():
    controller, runner = setup_runner()
    entered, release = asyncio.Event(), asyncio.Event()
    async def operation():
        entered.set()
        await release.wait()
    caller = asyncio.create_task(runner.run(request("a"), operation))
    await entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert controller.summary()["transfer_admission/active"] == 1
    release.set()
    await asyncio.gather(*runner.operations)
    assert controller.summary()["transfer_admission/active"] == 0


@pytest.mark.asyncio
async def test_only_confirmed_backend_failure_releases_budget():
    controller, runner = setup_runner()
    async def settled():
        raise SettledRestoreFailure("GET completed with missing object")
    with pytest.raises(SettledRestoreFailure):
        await runner.run(request("settled"), settled)
    assert controller.summary()["transfer_admission/active"] == 0
    async def unknown():
        raise TimeoutError("backend completion unknown")
    with pytest.raises(TimeoutError):
        await runner.run(request("unknown"), unknown)
    assert controller.summary()["transfer_admission/active"] == 1


@pytest.mark.asyncio
async def test_wrong_physical_resource_domain_is_rejected():
    _, runner = setup_runner()
    wrong = TransferRequest("wrong", TransferIntent.REMOTE_KV_RESTORE, 60,
        cost_gate_approved=True, metadata={"resource_domain": "nic-b"})
    with pytest.raises(ValueError, match="resource domain"):
        await runner.run(wrong, lambda: asyncio.sleep(0))


@pytest.mark.asyncio
async def test_disable_stops_queue_but_does_not_cancel_active_io():
    controller, runner = setup_runner()
    entered, release = asyncio.Event(), asyncio.Event()
    async def active():
        entered.set()
        await release.wait()
    a = asyncio.create_task(runner.run(request('active'), active))
    await entered.wait()
    async def never():
        pytest.fail('disabled queued operation started')
    b = asyncio.create_task(runner.run(request('queued'), never))
    await asyncio.sleep(0.01)
    state = await runner.disable(wait_timeout_seconds=0)
    assert state['pending_operations'] >= 1
    assert controller.summary()['transfer_admission/active'] == 1
    assert (await b).reason == 'disabled'
    assert not (await runner.run(request('new'), never)).used_optional_io
    release.set()
    await a
    state = await runner.disable()
    assert state['pending_operations'] == 0 and not state['unsettled_transfer_ids']
    assert controller.summary()['transfer_admission/active'] == 0


@pytest.mark.asyncio
async def test_disable_reports_unknown_transport_instead_of_faking_release():
    controller, runner = setup_runner()
    async def unknown():
        raise TimeoutError('native completion unconfirmed')
    with pytest.raises(TimeoutError):
        await runner.run(request('unknown'), unknown)
    state = await runner.disable()
    assert state['pending_operations'] == 0
    assert state['unsettled_transfer_ids'] == ['unknown']
    assert controller.summary()['transfer_admission/active'] == 1


@pytest.mark.asyncio
async def test_disable_keeps_uncertain_release_visible_when_admit_returns_late():
    _, runner = setup_runner()
    entered, reply = asyncio.Event(), asyncio.Event()
    async def begin(_):
        entered.set()
        await reply.wait()
        return {'action': 'admit', 'shadow': False}
    async def complete(_):
        raise ConnectionError('completion authority unavailable')
    runner.begin, runner.complete = begin, complete
    async def forbidden():
        pytest.fail('operation launched after disable')
    caller = asyncio.create_task(runner.run(request('late'), forbidden))
    await entered.wait()
    await runner.disable(0)
    reply.set()
    with pytest.raises(ConnectionError):
        await caller
    assert (await runner.disable())['unsettled_transfer_ids'] == ['late']
