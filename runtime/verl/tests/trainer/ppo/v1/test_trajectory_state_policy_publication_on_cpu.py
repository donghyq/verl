# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""CPU fault/ordering tests. Engine observations below are explicit fixtures.

No model, GPU, Ray service, live engine or optimizer is started by this module.
These tests cannot establish that a real checkpoint/Gateway integration called
the hooks; that remains the integration owner's separate validation.
"""

import asyncio
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from test_trajectory_state_runtime_on_cpu import (
    IDENTITY, AsyncClient, input_kwargs, mixed_source, make_runtime,
    snapshot as runtime_snapshot, wait_for,
)
from test_trajectory_state_tool_agent_on_cpu import (
    checkpointed_source, identity as tool_identity, make_loop, owner_ok, state_args, token_output,
)
from verl.trainer.ppo.v1.trajectory_state import (
    DurableTrajectoryCoordinator, EnginePolicyObservation, PolicyPublicationBlocked,
    PolicyPublicationDriver, PublishedPolicyRecord,
)
from verl.trainer.ppo.v1.trajectory_state.preemptible_tool_agent_loop import _generation_segments
from verl.trainer.ppo.v1.trajectory_state.preemptible_tool_agent_loop import TrajectoryCheckpointYield


ENGINES = ["http://owned-a:30000", "http://owned-b:30000"]


def version_identity(version):
    return {**IDENTITY, "policy_version": str(version), "model_weights_fingerprint": f"fixture-weights-{version}"}


def observations(version):
    return [{"engine_url": url, "weight_version": str(version), "source": "model_info"} for url in ENGINES]


def committed(coordinator, identity):
    ticket = coordinator.begin_policy_publication(identity, "fixture-static-model-config", "original-checkpoint", ENGINES)
    coordinator.mark_policy_update_started(ticket)
    return coordinator.confirm_policy_publication(ticket, observations(identity["policy_version"]))


def test_actual_transfer_readback_durability_precede_original_admission_once(tmp_path):
    async def scenario():
        c = DurableTrajectoryCoordinator(str(tmp_path), "run")
        events = []
        try:
            async def stop():
                events.append("original-stop")
            async def update(version):
                events.append(("original-update", version))
                with pytest.raises(PolicyPublicationBlocked):
                    c.get_published_policy(version)
            async def read():
                events.append("actual-engine-read")
                return observations(1)
            async def start(version):
                assert c.get_published_policy(version)["identity"] == version_identity(1)
                events.append(("original-start", version))
            record = await PolicyPublicationDriver(AsyncClient(c)).publish(
                identity=version_identity(1), model_configuration_fingerprint="fixture-static-model-config",
                publisher_id="original-checkpoint", engine_urls=ENGINES, update_checkpoint=update,
                observe_engines=read, stop_admission=stop, start_admission=start)
            assert events == ["original-stop", ("original-update", 1), "actual-engine-read", ("original-start", 1)]
            with pytest.raises(FrozenInstanceError):
                record.identity.policy_version = "999"
            raw = c.get_published_policy(1)
            raw["identity"]["policy_version"] = "999"
            assert c.get_published_policy(1)["identity"]["policy_version"] == "1"
            with pytest.raises(PolicyPublicationBlocked, match="already published"):
                committed(c, version_identity(1))
        finally:
            c.close()
        reopened = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            assert PublishedPolicyRecord.decode(reopened.get_active_published_policy()) == record
        finally:
            reopened.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["update", "read", "missing", "mismatch", "duplicate", "unexpected"])
def test_failed_or_incomplete_publication_never_registers_or_authorizes_version(tmp_path, fault):
    async def scenario():
        c = DurableTrajectoryCoordinator(str(tmp_path), "run")
        started = []
        try:
            async def stop():
                return None
            async def update(version):
                if fault == "update":
                    raise RuntimeError("actual transfer failed")
            async def read():
                if fault == "read":
                    raise RuntimeError("actual observation unavailable")
                rows = observations(2)
                if fault == "missing":
                    rows.pop()
                elif fault == "mismatch":
                    rows[-1]["weight_version"] = "1"
                elif fault == "duplicate":
                    rows.append(rows[0])
                elif fault == "unexpected":
                    rows[-1]["engine_url"] = "http://unowned:30000"
                return rows
            async def start(version):
                started.append(version)
            with pytest.raises((RuntimeError, ValueError)):
                await PolicyPublicationDriver(AsyncClient(c)).publish(
                    identity=version_identity(2), model_configuration_fingerprint="fixture-static-model-config",
                    publisher_id="original-checkpoint", engine_urls=ENGINES, update_checkpoint=update,
                    observe_engines=read, stop_admission=stop, start_admission=start)
            assert started == [] and c.summary()["policy_publication_blocked"]
            with pytest.raises(PolicyPublicationBlocked):
                c.get_published_policy(2)
            with pytest.raises(PolicyPublicationBlocked):
                committed(c, version_identity(3))
        finally:
            c.close()
    asyncio.run(scenario())


def test_hook_api_sits_inside_existing_lifecycle_and_failed_registration_does_not_repeat_transfer(tmp_path):
    async def scenario():
        c = DurableTrajectoryCoordinator(str(tmp_path), "run")
        hooks = PolicyPublicationDriver(AsyncClient(c))
        try:
            ticket = await hooks.prepare(identity=version_identity(1),
                model_configuration_fingerprint="fixture-static-model-config", publisher_id="original-checkpoint",
                engine_urls=ENGINES)
            # Original lifecycle removes/pauses its own engines here.
            await hooks.mark_update_started(ticket)
            # Original checkpoint transfer/set_global_steps completes here.
            async def actual_read():
                return observations(1)
            record = await hooks.confirm_before_admission(ticket, actual_read)
            # Original register fails after commit. It owns cleanup and remains
            # closed; the module does not rerun transfer to fix registration.
            assert record.identity.policy_version == "1"
            with pytest.raises(PolicyPublicationBlocked, match="already published"):
                await hooks.prepare(identity=version_identity(1),
                    model_configuration_fingerprint="fixture-static-model-config", publisher_id="original-checkpoint",
                    engine_urls=ENGINES)
        finally:
            c.close()
    asyncio.run(scenario())


def test_restart_during_weight_mutation_rejects_old_ticket_and_unknown_version(tmp_path):
    c = DurableTrajectoryCoordinator(str(tmp_path), "run")
    committed(c, version_identity(1))
    ticket = c.begin_policy_publication(version_identity(2), "fixture-static-model-config", "original-checkpoint", ENGINES)
    c.mark_policy_update_started(ticket)
    c.close()
    reopened = DurableTrajectoryCoordinator(str(tmp_path), "run")
    try:
        assert reopened.get_published_policy(1)["identity"] == version_identity(1)
        unsettled = reopened.get_policy_coordination_state()["publication"]
        assert unsettled["operation_id"] == ticket.operation_id and unsettled["state"] == "BLOCKED"
        assert unsettled["failure"] == "COORDINATOR_RESTART_FROM_UPDATING"
        with pytest.raises(PolicyPublicationBlocked):
            reopened.get_active_published_policy()
        with pytest.raises(PolicyPublicationBlocked):
            reopened.confirm_policy_publication(ticket, observations(2))
        with pytest.raises(PolicyPublicationBlocked):
            committed(reopened, version_identity(3))
    finally:
        reopened.close()


def test_uncertain_commit_after_atomic_replace_keeps_success_and_does_not_repeat_weights(tmp_path):
    armed = [False]
    def fail(stage):
        if armed[0] and stage == "after_replace":
            raise OSError("fixture lost acknowledgement after atomic replace")
    c = DurableTrajectoryCoordinator(str(tmp_path), "run", fault_hook=fail)
    ticket = c.begin_policy_publication(version_identity(1), "fixture-static-model-config", "original-checkpoint", ENGINES)
    c.mark_policy_update_started(ticket)
    armed[0] = True
    with pytest.raises(OSError):
        c.confirm_policy_publication(ticket, observations(1))
    c.close()
    reopened = DurableTrajectoryCoordinator(str(tmp_path), "run")
    try:
        assert reopened.get_active_published_policy()["identity"] == version_identity(1)
        assert reopened.get_policy_coordination_state()["publication"] is None
        with pytest.raises(PolicyPublicationBlocked, match="already published"):
            committed(reopened, version_identity(1))
    finally:
        reopened.close()


@pytest.mark.parametrize("changed", ["model", "template", "schema", "rewind"])
def test_publication_cannot_change_model_configuration_or_rewind_steps(tmp_path, changed):
    c = DurableTrajectoryCoordinator(str(tmp_path), "run")
    try:
        committed(c, version_identity(2))
        candidate = version_identity(3)
        lineage = "fixture-static-model-config"
        if changed == "model":
            lineage = "unrelated-model"
        elif changed == "template":
            candidate["tokenizer_template_fingerprint"] = "different-template"
        elif changed == "schema":
            candidate["kv_cache_schema_fingerprint"] = "different-kv"
        else:
            candidate["policy_version"] = "1"
        with pytest.raises(ValueError):
            c.begin_policy_publication(candidate, lineage, "original-checkpoint", ENGINES)
    finally:
        c.close()


def test_bounded_cut_requires_real_source_rights_and_timeout_does_not_open_publication(tmp_path):
    mono, wall = [100.0], [1700000000.0]
    c = DurableTrajectoryCoordinator(str(tmp_path), "run", clock=lambda: mono[0], wall_clock=lambda: wall[0])
    try:
        committed(c, version_identity(1))
        lease = c.open_trajectory("group/member-0", "source", 60)
        c.register_owned_running_trajectory(lease)
        cut = c.begin_handoff_policy_barrier("group", [lease], 5)
        assert c.check_handoff_policy_barrier(cut) == version_identity(1)
        with pytest.raises(PolicyPublicationBlocked):
            committed(c, version_identity(2))
        mono[0] += 6
        with pytest.raises(PolicyPublicationBlocked, match="expired"):
            c.check_handoff_policy_barrier(cut)
        with pytest.raises(PolicyPublicationBlocked):
            committed(c, version_identity(2))
        with pytest.raises(PolicyPublicationBlocked, match="rights"):
            c.release_handoff_policy_barrier(cut)
        c.abort_owned_trajectory(lease)
        c.release_handoff_policy_barrier(cut)  # Explicitly fenced failure cleanup.
        committed(c, version_identity(2))
    finally:
        c.close()


def test_restart_invalidates_cut_deadline_and_old_execution_rights(tmp_path):
    c = DurableTrajectoryCoordinator(str(tmp_path), "run")
    committed(c, version_identity(1))
    lease = c.open_trajectory("group/member-0", "source", 60)
    cut = c.begin_handoff_policy_barrier("group", [lease], 10)
    c.close()
    reopened = DurableTrajectoryCoordinator(str(tmp_path), "run", clock=lambda: 1)
    try:
        with pytest.raises(ValueError, match="stale"):
            reopened.check_trajectory_owner(lease)
        with pytest.raises(PolicyPublicationBlocked):
            reopened.check_handoff_policy_barrier(cut)
        with pytest.raises(PolicyPublicationBlocked):
            committed(reopened, version_identity(2))
        reopened.release_handoff_policy_barrier(cut)  # Old source can no longer commit.
        committed(reopened, version_identity(2))
    finally:
        reopened.close()


def test_runtime_keeps_manifest_source_identity_and_new_target_requires_explicit_token_only(tmp_path):
    async def scenario():
        c = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            committed(c, IDENTITY)
            client = AsyncClient(c)
            source = make_runtime(c, published_policy_client=client)
            result, _ = await mixed_source(source)
            assert not c.summary()["handoff_policy_barrier"]
            original = c.get_handoff_manifest(result.handoff_id)
            assert original.policy_identity == IDENTITY
            committed(c, version_identity(2))
            target = make_runtime(c, owner="target", published_policy_client=client)
            called = []
            async def resume(index, saved, kwargs, lease, check_owner):
                await check_owner()
                assert saved.policy_identity == IDENTITY and saved.agent_state["response_logprobs"] == [-.5, 0., 0.]
                assert check_owner.token_only_resume and check_owner.target_policy_identity == version_identity(2)
                assert kwargs == input_kwargs(8)[index]
                called.append(index)
                return {"member_index": index}
            async def pending(index, kwargs, lease, check_owner):
                await check_owner()
                assert kwargs == input_kwargs(8)[index]
                called.append(index)
                return {"member_index": index}
            with pytest.raises(RuntimeError):
                await target.run_target(result.handoff_id, resume, pending)
            assert called == []
            completed = await target.run_target(result.handoff_id, resume, pending, token_only=True)
            assert completed.member_kwargs == input_kwargs(8)
            assert called == [2, 3, 4, 5, 6, 7]
            assert asdict(c.get_handoff_manifest(result.handoff_id)) == asdict(original)
            await target.run_target(result.handoff_id, resume, pending, token_only=True)
            assert called == [2, 3, 4, 5, 6, 7]  # Existing outcomes are retained.
        finally:
            c.close()
    asyncio.run(scenario())


def test_real_tool_adapter_records_two_attempt_versions_without_rewriting_behavior_logprobs(tmp_path):
    async def scenario():
        c = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            committed(c, tool_identity(1))
            committed(c, tool_identity(2))
            output = token_output((41, 42))
            output.extra_fields = {"min_global_steps": 1, "max_global_steps": 2, "global_steps": 2,
                "native_gateway_attempts": [
                    {"output_start": 0, "output_end": 1, "native_meta_info": {
                        "weight_version": "1", "weight_versions": [{"start": 0, "end": 1, "version": "1"}]}},
                    {"output_start": 1, "output_end": 2, "native_meta_info": {
                        "weight_version": "2", "weight_versions": [{"start": 0, "end": 1, "version": "2"}]}},
                ]}
            async def resolve(version):
                return c.get_published_policy(version)
            loop = make_loop([output], [])
            args = {**state_args(), "global_steps": 77}  # Prompt birth is not the engine version.
            result = await loop.run({"logprobs": True}, _trajectory_check_owner=owner_ok,
                                    _trajectory_policy_resolver=resolve, **args)
            assert result.response_ids == [41, 42] and result.response_logprobs == [-.5, -.6]
            segments = result.extra_fields["trajectory_segments"]
            assert [(row["start"], row["end"], row["policy_version"]) for row in segments] == [(0, 1, "1"), (1, 2, "2")]
            assert [row["policy_identity"] for row in segments] == [tool_identity(1), tool_identity(2)]
            assert args["global_steps"] == 77  # No rewrite of original ReplayBuffer birth step.
            with pytest.raises(PolicyPublicationBlocked):
                await resolve(3)
            with pytest.raises(ValueError):
                _generation_segments(output, {}, tool_identity(2))  # Default remains single-policy fail-closed.
        finally:
            c.close()
    asyncio.run(scenario())


def test_source_cut_uses_current_publication_without_relabeling_old_generated_history(tmp_path):
    async def scenario():
        c = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            committed(c, IDENTITY)
            committed(c, version_identity(2))
            client = AsyncClient(c)
            source = make_runtime(c, published_policy_client=client)
            release = asyncio.Event()
            original = input_kwargs(8)
            for row in original:
                row["global_steps"] = 1
            async def run_member(index, kwargs, lease, commit, check):
                await check()
                if index < 2:
                    return {"member_index": index}
                await release.wait()
                cut_identity = await commit.get_snapshot_policy_identity()
                saved = runtime_snapshot("g", index, 8, lease, kwargs)
                saved.agent_state["extra_fields"]["trajectory_segments"][0]["policy_identity"] = IDENTITY
                saved = replace(saved, policy_identity=cut_identity)
                assert saved.agent_state["extra_fields"]["trajectory_segments"][0]["policy_version"] == "1"
                assert cut_identity == version_identity(2)
                reference = await commit(saved)
                raise TrajectoryCheckpointYield(saved, reference)
            task = asyncio.create_task(source.run_source("g", original, run_member, start_members=list(range(6))))
            await wait_for(lambda: source.group_status("g") is not None
                           and source.group_status("g")["completed_members"] == [0, 1]
                           and source.group_status("g")["running_members"] == [2, 3, 4, 5])
            await source.request_preemption("g")
            with pytest.raises(PolicyPublicationBlocked):
                committed(c, version_identity(3))
            release.set()
            result = await task
            assert c.get_handoff_manifest(result.handoff_id).policy_identity == version_identity(2)
            assert result.member_kwargs == original and all(row["global_steps"] == 1 for row in result.member_kwargs)
            assert not c.summary()["handoff_policy_barrier"]
            committed(c, version_identity(3))
        finally:
            c.close()
    asyncio.run(scenario())


def test_token_only_adapter_preserves_source_history_and_does_not_send_kv_reference(tmp_path):
    async def scenario():
        c = DurableTrajectoryCoordinator(str(tmp_path), "run")
        try:
            committed(c, tool_identity(1))
            committed(c, tool_identity(2))
            _, _, saved = await checkpointed_source()
            before = saved.encode()
            async def check():
                return None
            async def resolve(version):
                return c.get_published_policy(version)
            check.token_only_resume, check.target_policy_identity = True, tool_identity(2)
            check.resolve_published_policy = resolve
            calls = []
            target = make_loop([token_output((41, 42), version=2)], calls)
            server = target.server_manager
            output = await target.resume_from_snapshot(saved, check_owner=check, execution_generation=1,
                environment_version="env-v1", tool_version="tool-v1")
            assert saved.encode() == before and calls == []
            assert target.tokenizer.calls == 0
            assert server.calls[0]["prompt_ids"] == saved.agent_state["prompt_ids"]
            assert "kv_ref" not in server.calls[0]
            assert output.response_logprobs[:-2] == saved.agent_state["response_logprobs"]
            assert output.extra_fields["trajectory_resume_mode"] == "token_only"
            assert [row["policy_version"] for row in output.extra_fields["trajectory_segments"]] == ["1", "2"]
        finally:
            c.close()
    asyncio.run(scenario())
