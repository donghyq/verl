"""CPU checks of observations that must not fail after a committed GPU update."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def harness(monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "project-b-final-validation-20260928/terminal-20261007"
    monkeypatch.syspath_prepend(str(directory))
    monkeypatch.setenv("ROLLOUT_NAME", "sglang")
    import gate1_cluster_harness
    return gate1_cluster_harness


def test_real_parameter_observation_uses_same_tensor_not_loss_or_learning_rate(harness, monkeypatch):
    monkeypatch.setattr(harness, "placement", lambda: {"scope": "CPU fixture"})
    model = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
    worker = object.__new__(harness.ObservedDetachActorWorker)
    worker.actor = SimpleNamespace(engine=SimpleNamespace(module=model))
    before = worker.gate_capture_parameters()
    assert before["parameter_tensors"] == 1
    with torch.no_grad():
        model.weight[0, 0] += 0.125
    result = worker.gate_parameter_delta()
    assert result["changed_elements"] == 1
    assert result["max_abs_delta"] > 0
    assert result["examples"][0]["before_digest"] != result["examples"][0]["after_digest"]
    assert worker._gate_before == {}


def test_nonfinite_parameter_delta_cannot_be_recorded_as_pass(harness, monkeypatch):
    monkeypatch.setattr(harness, "placement", lambda: {})
    model = torch.nn.Linear(2, 1, bias=False)
    worker = object.__new__(harness.ObservedDetachActorWorker)
    worker.actor = SimpleNamespace(engine=SimpleNamespace(module=model))
    worker.gate_capture_parameters()
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    with pytest.raises(AssertionError, match="non-finite"):
        worker.gate_parameter_delta()


def test_observation_json_accepts_real_numpy_counters_and_preserves_null(harness, tmp_path):
    target = tmp_path / "observed.json"
    harness.write_json(target, {"members": np.arange(8, dtype=np.int32), "gradient": np.float32(1.25),
                               "recomputed_prefill_tokens": None})
    observed = json.loads(target.read_text())
    assert observed["members"] == list(range(8))
    assert observed["gradient"] == 1.25
    assert observed["recomputed_prefill_tokens"] is None


def test_replay_detection_uses_committed_batch_boundary_not_same_search_arguments(harness):
    events = [
        {"kind": "tool_started", "phase": "B", "member_index": 2, "batch_index": 0},
        {"kind": "tool_started", "phase": "B", "member_index": 2, "batch_index": 1},
        {"kind": "tool_started", "phase": "B", "member_index": 6, "batch_index": 0},
    ]
    assert harness.tool_replay_count(events, {2: 1}) == 1


@pytest.mark.asyncio
async def test_handoff_only_reuses_input_and_never_dispatches_completed_baselines(harness, tmp_path, monkeypatch):
    from tests.experimental.fully_async_policy.test_rollouter_handoff_integration import request_batch

    class RemoteCall:
        def __init__(self, callback):
            self.callback = callback

        async def remote(self, *args, **kwargs):
            return self.callback(*args, **kwargs)

    observer = SimpleNamespace(state=RemoteCall(lambda arm: {"events": [
        {"kind": "claim_group_for_artifact", "claimed": True}, {"kind": "acknowledge"},
    ]}))
    monkeypatch.setattr(harness, "GateObserver", SimpleNamespace(remote=lambda _: observer))
    monkeypatch.setattr(harness, "placement", lambda: {"scope": "CPU driver-control fixture"})
    monkeypatch.setattr(harness, "gpu_inventory", lambda: {"rows": None, "scope": "CPU fixture"})
    arms, updates = [], []
    original = request_batch()
    payload = harness.encode_handoff_member_record(original)

    async def run_arm(components, observed, inputs, arm, output, timeout):
        assert observed is observer and arm == "handoff"
        assert inputs.non_tensor_batch["raw_prompt"].tolist() == original.non_tensor_batch["raw_prompt"].tolist()
        arms.append(arm)
        return {"mode": arm}

    async def already_consumed(*args, **kwargs):
        return None

    monkeypatch.setattr(harness, "run_arm", run_arm)
    monkeypatch.setattr(harness, "ObservedArtifactClient", lambda *args: SimpleNamespace(
        claim_group_for_artifact=already_consumed))
    components = {
        "rollouter": SimpleNamespace(),  # No gate_inputs method: re-generation is forbidden.
        "trainer": SimpleNamespace(gate_one_update=RemoteCall(lambda client: updates.append(client) or {})),
        "artifact_store": SimpleNamespace(get_status=RemoteCall(lambda sample_id: SimpleNamespace(value="consumed"))),
    }
    previous = tmp_path / "previous-baselines.json"
    previous.write_text('{"source":"r3","drain":true,"abort":true}\n')
    result = await harness.execute_gate(
        components, tmp_path, 300, handoff_only=True, frozen_input_payload=payload)
    assert arms == ["handoff"] and len(updates) == 1
    assert result["executed_arms"] == ["handoff"]
    assert result["baseline"] == [{"mode": "handoff"}]
    assert result["baseline_comparison_status"] == "NOT_PERFORMED_AFTER_REPAIR"
    assert result["previous_baselines_loaded"] is False
    assert previous.read_text() == '{"source":"r3","drain":true,"abort":true}\n'
    assert (tmp_path / "frozen-input.record").read_bytes() == payload
    plan = json.loads((tmp_path / "frozen-plan.json").read_text())
    assert plan["order"] == ["handoff"]
    assert plan["input_record_sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.asyncio
async def test_handoff_only_rejects_missing_original_inputs(harness, tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "GateObserver", SimpleNamespace(remote=lambda _: None))
    with pytest.raises(ValueError, match="frozen input record"):
        await harness.execute_gate({}, tmp_path, 300, handoff_only=True)
