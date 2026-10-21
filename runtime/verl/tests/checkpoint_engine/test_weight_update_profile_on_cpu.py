"""Run production coordination bodies with CPU RPC boundaries; no GPU claims."""

import ast
import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "checkpoint_profile_cpu", ROOT / "verl/checkpoint_engine/weight_update_profile.py"
)
if os.getenv("VERL_PROFILE_REAL_IMPORT") == "1":
    profile_module = importlib.import_module("verl.checkpoint_engine.weight_update_profile")
else:
    profile_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile_module)


def load_class_body(filename, class_name, namespace, methods=None):
    """Execute the saved class; substitute external CUDA/Ray imports only."""
    source = ROOT / filename
    if os.getenv("VERL_PROFILE_REAL_IMPORT") == "1":
        imported = importlib.import_module(filename.removesuffix(".py").replace("/", "."))
        assert Path(imported.__file__).resolve() == source.resolve()
        for name, value in namespace.items():
            setattr(imported, name, value)
        return getattr(imported, class_name)
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.decorator_list = []
    if methods is not None:
        cls.body = [
            node
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in methods
        ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[class_name]


class InjectedFailure(RuntimeError):
    pass


def publish(manager, global_steps):
    async def run():
        return await manager.update_weights(global_steps)

    return asyncio.run(run())


def make_manager(*, failure=None, gateway=False, backend="nccl"):
    calls = []
    error = InjectedFailure("original sync failure")

    def record(name):
        calls.append(name)
        if failure == name:
            raise error

    class Group:
        world_size = 2

        def __init__(self, role):
            self.role = role

        def execute_checkpoint_engine(self, method, **kwargs):
            name = method[0]
            record(self.role + ":" + name)
            return [SimpleNamespace(stage=name, result={})] * self.world_size

        def update_weights(self, **kwargs):
            record(self.role + ":update")
            result = {"checkpoint/changed_ratio": 0.25} if self.role == "actor" else {}
            return [SimpleNamespace(stage="transfer_apply", result=result)] * self.world_size

    class FakeRay:
        @staticmethod
        def get(refs):
            record("wait:" + refs[0].stage)
            return [ref.result for ref in refs]

    class Backend:
        @staticmethod
        def build_topology(actor_size, rollout_size, metadata):
            record("topology")
            return {"rank": list(range(actor_size))}, {"rank": list(range(actor_size, actor_size + rollout_size))}

    def worker_group(**kwargs):
        record("worker_group")
        return Group("rollout")

    namespace = {
        "asyncio": asyncio,
        "ray": FakeRay,
        "RayWorkerGroup": worker_group,
        "RayClassWithInitArgs": lambda **kwargs: None,
        "_worker_cls": object,
        "auto_await": lambda fn: fn,
        "WeightUpdateProfile": profile_module.WeightUpdateProfile,
        "weight_update_phase": profile_module.weight_update_phase,
    }
    cls = load_class_body("verl/checkpoint_engine/base.py", "CheckpointEngineManager", namespace)
    manager = cls.__new__(cls)
    manager.backend = backend
    manager.backend_cls = Backend
    manager.actor_wg = Group("actor")
    manager.replicas = [SimpleNamespace(workers=[object(), object()])]
    manager.gateway_control = None
    for method, name in (
        ("abort_replicas", "abort"),
        ("release_kv_cache_replicas", "release"),
        ("resume_kv_cache_replicas", "resume_kv"),
        ("resume_generation_replicas", "resume_generation"),
    ):

        async def operation(name=name):
            record(name)

        setattr(manager, method, operation)

    if gateway:
        control = SimpleNamespace()
        for method in (
            "prepare_policy_publication",
            "stop_admission",
            "mark_policy_update_started",
            "confirm_policy_publication_before_admission",
            "start_admission",
            "fail_policy_publication",
        ):

            async def operation(*args, method=method):
                record("gateway:" + method)
                return "ticket" if method == "prepare_policy_publication" else None

            setattr(control, method, operation)
        manager.gateway_control = control
    return manager, calls, error


def records(caplog):
    prefix = "WEIGHT_UPDATE_PROFILE "
    return [json.loads(row.getMessage().split(prefix, 1)[1]) for row in caplog.records if prefix in row.getMessage()]


def test_success_preserves_protocol_order_and_sender_metrics(caplog):
    manager, calls, _ = make_manager(gateway=True)
    metrics = publish(manager, 12)
    assert calls == [
        "gateway:prepare_policy_publication",
        "gateway:stop_admission",
        "gateway:mark_policy_update_started",
        "abort",
        "worker_group",
        "release",
        "actor:prepare",
        "rollout:prepare",
        "wait:prepare",
        "topology",
        "actor:init_process_group",
        "rollout:init_process_group",
        "wait:init_process_group",
        "actor:update",
        "rollout:update",
        "wait:transfer_apply",
        "actor:finalize",
        "rollout:finalize",
        "wait:finalize",
        "resume_kv",
        "resume_generation",
        "gateway:confirm_policy_publication_before_admission",
        "gateway:start_admission",
    ]
    assert metrics["checkpoint/changed_ratio"] == 0.25
    assert metrics["timing_s/checkpoint/total"] >= metrics["timing_s/checkpoint/process_group"] >= 0
    assert manager.last_weight_update_profile["status"] == "completed"
    assert {r["global_steps"] for r in records(caplog)} == {12}
    assert records(caplog)[-1]["status"] == "completed"


@pytest.mark.parametrize(
    ("failure", "phase"),
    [
        ("abort", "abort_requests"),
        ("worker_group", "create_worker_group"),
        ("release", "release_kv"),
        ("wait:prepare", "process_group.prepare"),
        ("topology", "process_group.topology"),
        ("wait:init_process_group", "process_group.init"),
        ("wait:transfer_apply", "transfer_apply"),
        ("wait:finalize", "finalize"),
        ("resume_kv", "resume_kv"),
        ("resume_generation", "resume_generation"),
        ("gateway:confirm_policy_publication_before_admission", "gateway.confirm_publication"),
    ],
)
def test_failure_keeps_original_error_and_never_reopens_admission(failure, phase, caplog):
    manager, calls, error = make_manager(failure=failure, gateway=True)
    with pytest.raises(InjectedFailure) as caught:
        publish(manager, 12)
    assert caught.value is error
    assert "gateway:start_admission" not in calls
    assert calls[-1] == "gateway:fail_policy_publication"
    receipt = manager.last_weight_update_profile
    assert receipt["status"] == "failed" and receipt["failed_phase"] == phase
    assert receipt["error_type"] == "InjectedFailure"
    assert records(caplog)[-1]["failed_phase"] == phase


def test_cancellation_during_transfer_is_propagated_and_not_readmitted(caplog):
    manager, calls, _ = make_manager(gateway=True)
    error = asyncio.CancelledError()

    async def cancelled_update(global_steps):
        with profile_module.weight_update_phase("transfer_apply"):
            raise error

    manager._update_weights = cancelled_update
    with pytest.raises(asyncio.CancelledError) as caught:
        publish(manager, 12)
    assert caught.value is error
    assert calls[-1] == "gateway:fail_policy_publication"
    assert "gateway:start_admission" not in calls
    assert manager.last_weight_update_profile["status"] == "cancelled"
    assert records(caplog)[-1]["status"] == "cancelled"


def test_invalid_version_does_not_start_publication():
    manager, calls, _ = make_manager(gateway=True)
    with pytest.raises(ValueError, match="committed global_steps"):
        publish(manager, None)
    assert calls == []
    assert manager.last_weight_update_profile["failed_phase"] == "gateway.validate_version"


def test_missing_optional_diagnostic_metadata_does_not_mask_original_error():
    manager, calls, error = make_manager(gateway=True)
    del manager.backend

    async def failed_update(global_steps):
        raise error

    manager._update_weights = failed_update
    with pytest.raises(InjectedFailure) as caught:
        publish(manager, 2)
    assert caught.value is error
    assert "gateway:start_admission" not in calls
    assert manager.last_weight_update_profile["backend"] is None


def test_naive_backend_preserves_original_colocated_path():
    manager, calls, _ = make_manager(backend="naive")
    metrics = publish(manager, 0)
    assert calls == ["actor:update", "wait:transfer_apply"]
    assert set(metrics) == {"timing_s/checkpoint/transfer_apply", "timing_s/checkpoint/total"}


def test_repeated_publications_have_independent_ids_and_metrics():
    manager, _, _ = make_manager()
    publish(manager, 0)
    first = manager.last_weight_update_profile
    publish(manager, 1)
    second = manager.last_weight_update_profile
    assert first["update_id"] != second["update_id"]
    assert first["global_steps"] == 0 and second["global_steps"] == 1
    assert set(first["durations_s"]) == set(second["durations_s"])


def test_diagnostics_failure_cannot_fail_sync(monkeypatch):
    def broken_logger(*args, **kwargs):
        raise OSError("closed output stream")

    monkeypatch.setattr(profile_module.logger, "info", broken_logger)
    manager, _, _ = make_manager()
    assert publish(manager, 3)["checkpoint/changed_ratio"] == 0.25
    assert manager.last_weight_update_profile["status"] == "completed"


def test_async_publications_do_not_share_phase_state():
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def first():
            with profile_module.WeightUpdateProfile("nccl", 1) as value:
                with profile_module.weight_update_phase("first_only"):
                    entered.set()
                    await release.wait()
            return value

        task = asyncio.create_task(first())
        await entered.wait()
        with profile_module.WeightUpdateProfile("nccl", 2) as second:
            with profile_module.weight_update_phase("second_only"):
                release.set()
        return await task, second

    first, second = asyncio.run(run())
    assert set(first.durations) == {"first_only", "total"}
    assert set(second.durations) == {"second_only", "total"}


@pytest.mark.parametrize("failure", [None, "collective_init", "barrier"])
def test_nccl_init_reports_worker_phase_without_changing_collectives(failure, caplog):
    calls = []
    error = InjectedFailure("collective failed")

    def record(name):
        calls.append(name)
        if failure == name:
            raise error

    namespace = {
        "CheckpointEngine": object,
        "WeightUpdateProfile": profile_module.WeightUpdateProfile,
        "logger": SimpleNamespace(info=lambda *args: None),
        "collective": SimpleNamespace(
            is_group_initialized=lambda group: False,
            init_collective_group=lambda *args: record("collective_init"),
            barrier=lambda group: record("barrier"),
        ),
    }
    cls = load_class_body(
        "verl/checkpoint_engine/nccl_checkpoint_engine.py",
        "NCCLCheckpointEngine",
        namespace,
        methods={"init_process_group"},
    )
    engine = cls.__new__(cls)
    engine.group_name = "test"
    engine.rebuild_group = False
    engine._connect_zmq_client = lambda metadata: record("metadata_connect")
    if failure:
        with pytest.raises(InjectedFailure) as caught:
            engine.init_process_group(2, 4, object(), 2)
        assert caught.value is error
        assert records(caplog)[-1]["failed_phase"] == failure
    else:
        engine.init_process_group(2, 4, object(), 2)
        assert calls == ["collective_init", "metadata_connect", "barrier"]
    receipt = records(caplog)[-1]
    assert receipt["operation"] == "init_process_group"
    assert receipt["details"] == {"rank": 2, "world_size": 4, "num_senders": 2, "group_name": "test"}


def test_summary_deduplicates_copied_logs_and_keeps_unfinished_time_unknown(caplog):
    with profile_module.WeightUpdateProfile("nccl", 7) as profile:
        with profile.phase("transfer_apply"):
            pass
    rows = records(caplog)
    completed = profile_module.summarize_profile_records(rows + rows)["profiles"][0]
    assert completed["event_count"] == len(rows)
    assert completed["total_s"] == profile.durations["total"]
    assert completed["open_phases"] == []
    incomplete = profile_module.summarize_profile_records(rows[:2])["profiles"][0]
    assert incomplete["status"] == "incomplete" and incomplete["total_s"] is None
    assert incomplete["open_phases"] == ["total", "transfer_apply"]


@pytest.mark.parametrize("elapsed", [-1, float("nan"), "1"])
def test_summary_rejects_invalid_measurements(elapsed, caplog):
    with profile_module.WeightUpdateProfile("nccl", 0):
        pass
    rows = records(caplog)
    rows[-1]["elapsed_s"] = elapsed
    with pytest.raises(ValueError, match="elapsed time"):
        profile_module.summarize_profile_records(rows)


def test_summary_rejects_conflicting_copies(caplog):
    with profile_module.WeightUpdateProfile("nccl", 0):
        pass
    rows = records(caplog)
    conflicting = {**rows[-1], "elapsed_s": rows[-1]["elapsed_s"] + 1}
    with pytest.raises(ValueError, match="Conflicting copies"):
        profile_module.summarize_profile_records(rows + [conflicting])


def test_missing_duration_remains_unknown(caplog):
    with profile_module.WeightUpdateProfile("nccl", 0):
        pass
    rows = records(caplog)
    rows[-1]["elapsed_s"] = None
    summary = profile_module.summarize_profile_records(rows)["profiles"][0]
    assert summary["status"] == "completed"
    assert summary["total_s"] is None and summary["durations_s"] == {}


def test_cli_summarizes_prefixed_logs_without_importing_serving_runtime(tmp_path, caplog):
    with profile_module.WeightUpdateProfile("nccl", 0):
        pass
    logfile = tmp_path / "driver.log"
    logfile.write_text("\n".join("(Driver pid=10) WEIGHT_UPDATE_PROFILE " + json.dumps(row) for row in records(caplog)))
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "verl/checkpoint_engine/weight_update_profile.py"),
            str(logfile),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text())
    assert summary["profiles"][0]["status"] == "completed"
    assert summary["wire_time_s"] is None and summary["gpu_time_s"] is None
