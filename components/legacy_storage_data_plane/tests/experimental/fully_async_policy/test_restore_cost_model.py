# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


_MODULE_PATH = Path(__file__).parents[3] / "verl/experimental/fully_async_policy/restore_cost_model.py"
_SPEC = importlib.util.spec_from_file_location("restore_cost_model_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

CostGatePath = _MODULE.CostGatePath
CostGateReason = _MODULE.CostGateReason
RecomputeCostModel = _MODULE.RecomputeCostModel
RestoreCostModel = _MODULE.RestoreCostModel
RestoreProfileSample = _MODULE.RestoreProfileSample
RestoreRecomputeBenchmark = _MODULE.RestoreRecomputeBenchmark
RestoreTimingBreakdown = _MODULE.RestoreTimingBreakdown
ShadowCostGateRecorder = _MODULE.ShadowCostGateRecorder
build_crossover_curve = _MODULE.build_crossover_curve
evaluate_restore_cost_gate = _MODULE.evaluate_restore_cost_gate
find_crossover_tokens = _MODULE.find_crossover_tokens

# A 25 GB/s RoCE link and a 3.5 MB/token KV footprint are the Project B
# modelling inputs; the fixed overhead stands for lookup, allocation and H2D
# import that the raw bandwidth number hides. Queueing is an EFT input, not a
# restore execution cost, so it must never be folded into the cost gate.
BANDWIDTH_BYTES_PER_S = 25e9
BYTES_PER_TOKEN = 3.5e6
PREFILL_TOKENS_PER_S = 800.0
FIXED_OVERHEAD_S = 0.5


@pytest.fixture
def restore_model():
    return RestoreCostModel(fixed_overhead_s=FIXED_OVERHEAD_S, bandwidth_bytes_per_s=BANDWIDTH_BYTES_PER_S)


@pytest.fixture
def recompute_model():
    return RecomputeCostModel(prefill_tokens_per_second=PREFILL_TOKENS_PER_S)


def make_sample(
    sample_id: str, missing_prefix_tokens: int, *, overhead_s: float = FIXED_OVERHEAD_S
) -> RestoreProfileSample:
    kv_bytes = int(missing_prefix_tokens * BYTES_PER_TOKEN)
    return RestoreProfileSample(
        sample_id=sample_id,
        policy_kv_identity_digest="identity-digest-a",
        missing_prefix_tokens=missing_prefix_tokens,
        kv_bytes=kv_bytes,
        timing=RestoreTimingBreakdown(
            lookup_s=0.0,
            allocation_s=overhead_s,
            transfer_s=kv_bytes / BANDWIDTH_BYTES_PER_S,
            import_s=0.0,
        ),
        recompute_latency_s=missing_prefix_tokens / PREFILL_TOKENS_PER_S,
        restored_tokens=missing_prefix_tokens,
    )


def test_sample_reports_restore_recompute_and_effective_bandwidth():
    sample = make_sample("s1", 1024)

    assert sample.t_restore_s == pytest.approx(FIXED_OVERHEAD_S + 1024 * BYTES_PER_TOKEN / BANDWIDTH_BYTES_PER_S)
    assert sample.t_recompute_s == pytest.approx(1024 / PREFILL_TOKENS_PER_S)
    assert sample.effective_bandwidth_bytes_per_s == pytest.approx(BANDWIDTH_BYTES_PER_S)
    assert sample.measured_prefill_tokens_per_second == pytest.approx(PREFILL_TOKENS_PER_S)


def test_sample_from_metrics_accepts_bandwidth_instead_of_transfer_latency():
    sample = RestoreProfileSample.from_metrics(
        {
            "sample_id": "s-bandwidth",
            "policy_kv_identity_digest": "identity-digest-a",
            "missing_prefix_tokens": 1000,
            "kv_bytes": 10_000_000,
            "lookup_latency_s": 0.01,
            "queue_latency_s": 0.02,
            "allocation_latency_s": 0.03,
            "effective_bandwidth_bytes_per_s": 1e9,
            "import_latency_s": 0.04,
            "actual_prefill_latency_s": 2.0,
            "restored_tokens": 900,
        }
    )

    assert sample.timing.transfer_s == pytest.approx(0.01)
    assert sample.timing.non_transfer_s == pytest.approx(0.08)
    assert sample.t_restore_s == pytest.approx(0.09)
    assert sample.t_restore_e2e_s == pytest.approx(0.11)
    assert sample.net_savings_s == pytest.approx(1.91)


def test_queue_time_is_recorded_but_excluded_from_restore_cost_gate():
    sample = RestoreProfileSample(
        sample_id="queued-restore",
        policy_kv_identity_digest="identity-digest-a",
        missing_prefix_tokens=1024,
        kv_bytes=1024,
        timing=RestoreTimingBreakdown(
            queue_s=4.0,
            allocation_s=0.2,
            transfer_s=0.1,
            import_s=0.1,
        ),
        recompute_latency_s=1.0,
        restored_tokens=1024,
    )

    assert sample.t_restore_s == pytest.approx(0.4)
    assert sample.t_restore_e2e_s == pytest.approx(4.4)
    assert sample.net_savings_s == pytest.approx(0.6)

    restore_model = RestoreCostModel.calibrate([sample])
    assert restore_model.fixed_overhead_s == pytest.approx(0.4)
    assert restore_model.t_restore_s(kv_bytes=1024) == pytest.approx(0.4)


def test_sample_from_metrics_requires_a_transfer_measurement():
    with pytest.raises(ValueError, match="either remote_transfer_latency_s or a positive effective_bandwidth"):
        RestoreProfileSample.from_metrics(
            {
                "sample_id": "s-missing",
                "policy_kv_identity_digest": "identity-digest-a",
                "missing_prefix_tokens": 10,
                "kv_bytes": 100,
                "actual_prefill_latency_s": 1.0,
            }
        )


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"sample_id": ""}, "sample_id must be non-empty"),
        ({"policy_kv_identity_digest": ""}, "policy_kv_identity_digest must be non-empty"),
        ({"missing_prefix_tokens": -1}, "missing_prefix_tokens must be a non-negative integer"),
        ({"kv_bytes": -1}, "kv_bytes must be a non-negative integer"),
        ({"restored_tokens": 4000}, "restored_tokens cannot exceed missing_prefix_tokens"),
        ({"recompute_latency_s": 0.0}, "recompute_latency_s must be positive"),
    ],
)
def test_sample_rejects_invalid_measurements(kwargs, expected_message):
    base = {
        "sample_id": "s",
        "policy_kv_identity_digest": "identity-digest-a",
        "missing_prefix_tokens": 1024,
        "kv_bytes": 1024,
        "timing": RestoreTimingBreakdown(),
        "recompute_latency_s": 1.0,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=expected_message):
        RestoreProfileSample(**base)


def test_timing_breakdown_rejects_negative_components():
    with pytest.raises(ValueError, match="queue_s must be non-negative"):
        RestoreTimingBreakdown(queue_s=-0.1)


def test_timing_breakdown_subtracts_observed_io_overlap():
    timing = RestoreTimingBreakdown(transfer_s=0.06, import_s=0.02, overlap_s=0.03)

    assert timing.execution_s == pytest.approx(0.05)
    assert timing.total_s == pytest.approx(0.05)


def test_calibration_recovers_measured_bandwidth_and_fixed_overhead():
    samples = [make_sample(f"s{index}", tokens) for index, tokens in enumerate([512, 1024, 2048, 4096])]

    restore_model = RestoreCostModel.calibrate(samples)
    recompute_model = RecomputeCostModel.calibrate(samples)

    assert restore_model.bandwidth_bytes_per_s == pytest.approx(BANDWIDTH_BYTES_PER_S)
    assert restore_model.fixed_overhead_s == pytest.approx(FIXED_OVERHEAD_S)
    assert restore_model.sample_count == 4
    assert recompute_model.prefill_tokens_per_second == pytest.approx(PREFILL_TOKENS_PER_S)
    assert recompute_model.fixed_overhead_s == pytest.approx(0.0)


def test_restore_calibration_accounts_for_residual_prefill():
    samples = [
        RestoreProfileSample(
            sample_id=f"residual-{tokens}",
            policy_kv_identity_digest="identity-digest-a",
            missing_prefix_tokens=tokens,
            kv_bytes=tokens * 100,
            timing=RestoreTimingBreakdown(
                transfer_s=tokens * 100 / 10_000,
                allocation_s=0.2,
                runtime_overhead_s=64 / 1_000,
            ),
            recompute_latency_s=0.1 + tokens / 1_000,
            restored_tokens=tokens - 64,
            residual_prefill_tokens=64,
        )
        for tokens in (256, 512, 1024)
    ]
    recompute_model = RecomputeCostModel.calibrate(samples)
    restore_model = RestoreCostModel.calibrate(
        samples,
        residual_prefill_seconds_per_token=recompute_model.incremental_seconds_per_token,
    )

    assert recompute_model.fixed_overhead_s == pytest.approx(0.1)
    assert recompute_model.prefill_tokens_per_second == pytest.approx(1_000)
    assert restore_model.fixed_overhead_s == pytest.approx(0.2)
    assert restore_model.t_restore_s(kv_bytes=102_400, residual_prefill_tokens=64) == pytest.approx(10.504)


def test_calibration_requires_samples():
    with pytest.raises(ValueError, match="at least one sample is required to calibrate a restore cost model"):
        RestoreCostModel.calibrate([])
    with pytest.raises(ValueError, match="at least one sample is required to calibrate a recompute cost model"):
        RecomputeCostModel.calibrate([])


def test_cost_gate_prefers_recompute_for_a_short_prefix(restore_model, recompute_model):
    decision = evaluate_restore_cost_gate(
        restore_model=restore_model,
        recompute_model=recompute_model,
        missing_prefix_tokens=64,
        kv_bytes=int(64 * BYTES_PER_TOKEN),
        identity_digest="identity-digest-a",
    )

    assert decision.path is CostGatePath.TOKEN_ONLY_REPREFILL
    assert decision.reason is CostGateReason.RESTORE_COST_EXCEEDS_RECOMPUTE
    assert decision.should_restore is False
    assert decision.net_savings_s < 0


def test_cost_gate_prefers_restore_for_a_long_prefix(restore_model, recompute_model):
    decision = evaluate_restore_cost_gate(
        restore_model=restore_model,
        recompute_model=recompute_model,
        missing_prefix_tokens=4096,
        kv_bytes=int(4096 * BYTES_PER_TOKEN),
        identity_digest="identity-digest-a",
    )

    assert decision.path is CostGatePath.REMOTE_RESTORE
    assert decision.reason is CostGateReason.RESTORE_CHEAPER_THAN_RECOMPUTE
    assert decision.should_restore is True
    assert decision.net_savings_s > 0
    assert decision.shadow is True
    assert decision.affects_routing is False


def test_safety_margin_can_flip_a_marginal_restore_to_recompute(restore_model, recompute_model):
    tokens = 600
    kv_bytes = int(tokens * BYTES_PER_TOKEN)

    without_margin = evaluate_restore_cost_gate(
        restore_model=restore_model,
        recompute_model=recompute_model,
        missing_prefix_tokens=tokens,
        kv_bytes=kv_bytes,
    )
    with_margin = evaluate_restore_cost_gate(
        restore_model=restore_model,
        recompute_model=recompute_model,
        missing_prefix_tokens=tokens,
        kv_bytes=kv_bytes,
        safety_margin_s=1.0,
    )

    assert without_margin.should_restore is True
    assert with_margin.should_restore is False
    assert with_margin.reason is CostGateReason.RESTORE_COST_EXCEEDS_RECOMPUTE
    assert with_margin.t_restore_s == pytest.approx(without_margin.t_restore_s)


def test_minimum_savings_threshold_is_reported_explicitly(restore_model, recompute_model):
    decision = evaluate_restore_cost_gate(
        restore_model=restore_model,
        recompute_model=recompute_model,
        missing_prefix_tokens=4096,
        kv_bytes=int(4096 * BYTES_PER_TOKEN),
        min_savings_s=1e6,
    )

    assert decision.path is CostGatePath.TOKEN_ONLY_REPREFILL
    assert decision.reason is CostGateReason.SAVINGS_BELOW_THRESHOLD
    assert decision.net_savings_s > 0


def test_empty_prefix_is_not_a_restore_candidate(restore_model, recompute_model):
    decision = evaluate_restore_cost_gate(
        restore_model=restore_model,
        recompute_model=recompute_model,
        missing_prefix_tokens=0,
        kv_bytes=0,
    )

    assert decision.path is CostGatePath.TOKEN_ONLY_REPREFILL
    assert decision.reason is CostGateReason.NO_REUSABLE_PREFIX
    assert decision.should_restore is False


def test_crossover_curve_is_monotone_and_matches_binary_search(restore_model, recompute_model):
    curve = build_crossover_curve(
        restore_model=restore_model,
        recompute_model=recompute_model,
        bytes_per_token=BYTES_PER_TOKEN,
        token_grid=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    )

    outcomes = [decision.should_restore for decision in curve]
    assert outcomes == sorted(outcomes), "a longer prefix must never lose its restore advantage"
    assert outcomes[0] is False
    assert outcomes[-1] is True

    crossover = find_crossover_tokens(
        restore_model=restore_model,
        recompute_model=recompute_model,
        bytes_per_token=BYTES_PER_TOKEN,
    )
    assert crossover is not None
    below = [decision for decision in curve if decision.missing_prefix_tokens < crossover]
    above = [decision for decision in curve if decision.missing_prefix_tokens >= crossover]
    assert all(not decision.should_restore for decision in below)
    assert all(decision.should_restore for decision in above)


def test_crossover_curve_can_preserve_a_fixed_residual_prefill(restore_model, recompute_model):
    curve = build_crossover_curve(
        restore_model=restore_model,
        recompute_model=recompute_model,
        bytes_per_token=BYTES_PER_TOKEN,
        token_grid=[128, 512, 1024],
        residual_prefill_tokens=64,
    )

    assert [decision.t_restore_s for decision in curve] == sorted(
        decision.t_restore_s for decision in curve
    )
    assert [decision.missing_prefix_tokens for decision in curve] == [128, 512, 1024]

    crossover = find_crossover_tokens(
        restore_model=restore_model,
        recompute_model=recompute_model,
        bytes_per_token=BYTES_PER_TOKEN,
        residual_prefill_tokens=64,
    )
    assert crossover is not None


def test_crossover_curve_rejects_negative_fixed_residual_prefill(restore_model, recompute_model):
    with pytest.raises(ValueError, match="residual_prefill_tokens must be non-negative"):
        build_crossover_curve(
            restore_model=restore_model,
            recompute_model=recompute_model,
            bytes_per_token=BYTES_PER_TOKEN,
            residual_prefill_tokens=-1,
        )


def test_crossover_is_unreachable_when_bandwidth_cannot_beat_prefill(recompute_model):
    # 1 MB/s is far below what is needed for a 3.5 MB/token prefix.
    slow_link = RestoreCostModel(fixed_overhead_s=0.5, bandwidth_bytes_per_s=1e6)

    assert (
        find_crossover_tokens(
            restore_model=slow_link,
            recompute_model=recompute_model,
            bytes_per_token=BYTES_PER_TOKEN,
            max_tokens=65536,
        )
        is None
    )


def test_crossover_curve_validates_inputs(restore_model, recompute_model):
    with pytest.raises(ValueError, match="bytes_per_token must be positive"):
        build_crossover_curve(restore_model=restore_model, recompute_model=recompute_model, bytes_per_token=0.0)
    with pytest.raises(ValueError, match="token_grid entries must be positive"):
        build_crossover_curve(
            restore_model=restore_model,
            recompute_model=recompute_model,
            bytes_per_token=BYTES_PER_TOKEN,
            token_grid=[0],
        )
    with pytest.raises(ValueError, match="token_grid must be sorted ascending"):
        build_crossover_curve(
            restore_model=restore_model,
            recompute_model=recompute_model,
            bytes_per_token=BYTES_PER_TOKEN,
            token_grid=[512, 128],
        )


def test_benchmark_exports_machine_readable_csv_and_json(tmp_path, restore_model, recompute_model):
    samples = tuple(make_sample(f"s{index}", tokens) for index, tokens in enumerate([512, 1024, 2048, 4096]))
    recorder = ShadowCostGateRecorder(label="unit-test-benchmark")
    for tokens in (128, 4096):
        recorder.record(
            restore_model=restore_model,
            recompute_model=recompute_model,
            missing_prefix_tokens=tokens,
            kv_bytes=int(tokens * BYTES_PER_TOKEN),
            identity_digest="identity-digest-a",
        )

    benchmark = recorder.export(samples=samples, restore_model=restore_model, recompute_model=recompute_model)

    csv_path = benchmark.write_csv(tmp_path / "samples.csv")
    decision_csv_path = benchmark.write_decisions_csv(tmp_path / "decisions.csv")
    json_path = benchmark.write_json(
        tmp_path / "benchmark.json",
        restore_model=restore_model,
        recompute_model=recompute_model,
        extra={"run_id": "unit-test"},
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert float(rows[0]["kv_bytes"]) == pytest.approx(512 * BYTES_PER_TOKEN)
    assert float(rows[0]["t_restore_s"]) > 0

    with decision_csv_path.open(newline="", encoding="utf-8") as handle:
        decision_rows = list(csv.DictReader(handle))
    assert [row["path"] for row in decision_rows] == ["token_only_reprefill", "remote_restore"]

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "unit-test"
    assert payload["shadow_mode"] is True
    assert payload["affects_routing"] is False
    assert payload["decision_count"] == 2
    assert payload["restore_decision_count"] == 1
    assert payload["models"]["restore"]["bandwidth_bytes_per_s"] == pytest.approx(BANDWIDTH_BYTES_PER_S)
    assert payload["models"]["recompute"]["prefill_tokens_per_second"] == pytest.approx(PREFILL_TOKENS_PER_S)
    assert all(decision["shadow"] is True for decision in payload["decisions"])
    assert all(decision["affects_routing"] is False for decision in payload["decisions"])


def test_shadow_recorder_summarizes_without_changing_routing(restore_model, recompute_model):
    recorder = ShadowCostGateRecorder()
    for tokens in (128, 256, 4096, 8192):
        recorder.record(
            restore_model=restore_model,
            recompute_model=recompute_model,
            missing_prefix_tokens=tokens,
            kv_bytes=int(tokens * BYTES_PER_TOKEN),
        )

    summary = recorder.summary()

    assert summary["rollout_restore_cost_gate/decisions"] == 4.0
    assert summary["rollout_restore_cost_gate/restore_decisions"] == 2.0
    assert summary["rollout_restore_cost_gate/restore_decision_ratio"] == pytest.approx(0.5)
    assert len(recorder.decisions) == 4
    assert all(decision.shadow and not decision.affects_routing for decision in recorder.decisions)


def test_shadow_recorder_summary_is_empty_without_decisions():
    assert ShadowCostGateRecorder().summary() == {}


def test_benchmark_requires_content():
    with pytest.raises(ValueError, match="at least one sample or decision"):
        RestoreRecomputeBenchmark()


def test_infinite_bandwidth_model_has_no_transfer_term():
    instant = RestoreCostModel(fixed_overhead_s=0.25, bandwidth_bytes_per_s=math.inf)

    assert instant.t_restore_s(kv_bytes=10**9) == pytest.approx(0.25)


def test_restore_shadow_recorder_feeds_transfer_admission_cost_gate(restore_model, recompute_model):
    from verl.experimental.fully_async_policy.transfer_admission import (
        AdmissionAction,
        AdmissionReason,
        TransferAdmissionConfig,
        TransferAdmissionController,
        TransferIntent,
    )

    admission = TransferAdmissionController(
        TransferAdmissionConfig(max_inflight_bytes=int(8192 * BYTES_PER_TOKEN))
    )
    recorder = ShadowCostGateRecorder(transfer_admission=admission)

    denied = recorder.record(
        restore_model=restore_model,
        recompute_model=recompute_model,
        missing_prefix_tokens=128,
        kv_bytes=int(128 * BYTES_PER_TOKEN),
        identity_digest="identity-short",
    )
    approved = recorder.record(
        restore_model=restore_model,
        recompute_model=recompute_model,
        missing_prefix_tokens=8192,
        kv_bytes=int(8192 * BYTES_PER_TOKEN),
        identity_digest="identity-long",
    )

    assert denied.should_restore is False
    assert approved.should_restore is True
    assert len(admission.records) == 2
    assert admission.records[0].intent is TransferIntent.REMOTE_KV_RESTORE
    assert admission.records[0].action is AdmissionAction.REJECT
    assert admission.records[0].reason is AdmissionReason.RESTORE_COST_GATE_DENIED
    assert admission.records[1].action is AdmissionAction.ADMIT
    assert all(not record.affects_execution for record in admission.records)
