"""Tests for the end-to-end training-step simulation (Phase E)."""

import os
import tempfile

import pytest

from verl.experimental.partial_rollout.e2e_simulation import (
    PartialRolloutSimulation,
    SimulationConfig,
    SimulationReport,
    run_default_simulation,
)


class TestSimulationBasics:
    def test_default_config_runs(self):
        sim = PartialRolloutSimulation()
        report = sim.run()
        assert report.total_rollouts == 4
        assert report.completed == 4
        assert report.cancelled == 0

    def test_report_summary_prints(self):
        report = run_default_simulation()
        text = report.summary()
        assert "Partial Rollout E2E Simulation Report" in text
        assert "KV-aware resumes" in text
        assert "token-only resumes" in text
        assert "recomputed" in text


class TestKVAwareVsTokenOnly:
    def test_pre_update_token_only_post_update_kv_aware(self):
        cfg = SimulationConfig(
            num_rollouts=4,
            pause_before_weight_update=2,
            gen_len=16,
        )
        sim = PartialRolloutSimulation(cfg)
        report = sim.run()
        assert report.token_only_resumes == 2
        assert report.kv_aware_resumes == 2
        assert report.recomputed_tokens_token_only == 32
        assert report.recomputed_tokens_kv_aware == 0
        assert report.kv_reuse_ratio == 0.5

    def test_all_token_only(self):
        cfg = SimulationConfig(
            num_rollouts=3,
            pause_before_weight_update=3,
            gen_len=10,
        )
        sim = PartialRolloutSimulation(cfg)
        report = sim.run()
        assert report.token_only_resumes == 3
        assert report.kv_aware_resumes == 0
        assert report.recomputed_tokens_token_only == 30
        assert report.recomputed_tokens_kv_aware == 0
        assert report.kv_reuse_ratio == 0.0

    def test_all_kv_aware(self):
        cfg = SimulationConfig(
            num_rollouts=3,
            pause_before_weight_update=0,
            gen_len=10,
        )
        sim = PartialRolloutSimulation(cfg)
        report = sim.run()
        assert report.token_only_resumes == 0
        assert report.kv_aware_resumes == 3
        assert report.total_recomputed_tokens == 0
        assert report.kv_reuse_ratio == 1.0


class TestMetricsReport:
    def test_save_and_resume_latency_recorded(self):
        sim = PartialRolloutSimulation()
        report = sim.run()
        assert report.save_latency >= 0
        assert report.resume_latency >= 0

    def test_total_recomputed_tokens(self):
        cfg = SimulationConfig(
            num_rollouts=4,
            pause_before_weight_update=1,
            gen_len=8,
        )
        sim = PartialRolloutSimulation(cfg)
        report = sim.run()
        assert report.total_recomputed_tokens == 8
        assert report.recomputed_tokens_token_only == 8
        assert report.recomputed_tokens_kv_aware == 0


class TestDiskPersistence:
    def test_run_with_disk_persistence(self, tmp_path):
        cfg = SimulationConfig(num_rollouts=4, pause_before_weight_update=2)
        sim = PartialRolloutSimulation(cfg)
        report = sim.run_with_disk_persistence(str(tmp_path))
        assert report.store_root == str(tmp_path)
        assert report.completed == 4
        # store is cleared after run
        assert os.listdir(str(tmp_path)) == []


class TestManagerStateConsistency:
    def test_all_rollouts_completed(self):
        sim = PartialRolloutSimulation()
        sim.run()
        for i in range(4):
            state = sim.manager.get_state(f"sim-{i}")
            assert state is not None
            assert state.lifecycle.value == "completed"

    def test_no_active_after_run(self):
        sim = PartialRolloutSimulation()
        sim.run()
        assert sim.manager.active_count == 0
        assert sim.manager.metrics.complete_count == 4
        assert sim.manager.metrics.cancel_count == 2
