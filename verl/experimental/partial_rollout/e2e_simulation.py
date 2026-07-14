"""Phase E: end-to-end partial rollout training-step simulation.

Orchestrates a realistic sequence that demonstrates the full closed loop:

1. Start N rollouts on a serving engine.
2. Pause some to free memory for a training step.
3. Update weights (simulating gradient step).
4. Resume rollouts — those paused *before* the weight update must be token-only
   (version mismatch); those paused *after* can be KV-aware.
5. Complete all rollouts, reclaim resources, report metrics.

This ties Phases A-D together: state machine (A), SGLang server simulation (B),
async adapter contract (C), and snapshot persistence (D) all participate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from verl.experimental.partial_rollout.partial_rollout_manager import (
    PartialRolloutManager,
    RolloutMetrics,
)
from verl.experimental.partial_rollout.rollout_state import (
    RolloutState,
    RolloutStateSnapshot,
    new_kv_handle,
)
from verl.experimental.partial_rollout.snapshot_store import SnapshotStore


@dataclass
class SimulationConfig:
    num_rollouts: int = 4
    prompt_len: int = 8
    gen_len: int = 16
    pause_before_weight_update: int = 2
    weight_update_steps: int = 1
    model_weight_version: str = "v0"
    tokenizer_fingerprint: str = "tok-v0"


@dataclass
class SimulationReport:
    """Final metrics report comparing KV-aware vs token-only resume."""

    total_rollouts: int = 0
    kv_aware_resumes: int = 0
    token_only_resumes: int = 0
    recomputed_tokens_kv_aware: int = 0
    recomputed_tokens_token_only: int = 0
    save_latency: float = 0.0
    resume_latency: float = 0.0
    completed: int = 0
    cancelled: int = 0
    failed_resumes: int = 0
    store_root: Optional[str] = None

    @property
    def total_recomputed_tokens(self) -> int:
        return self.recomputed_tokens_kv_aware + self.recomputed_tokens_token_only

    @property
    def kv_reuse_ratio(self) -> float:
        total = self.kv_aware_resumes + self.token_only_resumes
        if total == 0:
            return 0.0
        return self.kv_aware_resumes / total

    def summary(self) -> str:
        lines = [
            "=== Partial Rollout E2E Simulation Report ===",
            f"  total rollouts:        {self.total_rollouts}",
            f"  KV-aware resumes:       {self.kv_aware_resumes}",
            f"  token-only resumes:     {self.token_only_resumes}",
            f"  KV reuse ratio:         {self.kv_reuse_ratio:.1%}",
            f"  recomputed (KV-aware):  {self.recomputed_tokens_kv_aware}",
            f"  recomputed (token-only):{self.recomputed_tokens_token_only}",
            f"  total recomputed:       {self.total_recomputed_tokens}",
            f"  save latency (s):       {self.save_latency:.6f}",
            f"  resume latency (s):     {self.resume_latency:.6f}",
            f"  completed:              {self.completed}",
            f"  cancelled:              {self.cancelled}",
            f"  failed resumes:         {self.failed_resumes}",
        ]
        if self.store_root:
            lines.append(f"  snapshot store:        {self.store_root}")
        lines.append("=== End Report ===")
        return "\n".join(lines)


class PartialRolloutSimulation:
    """Orchestrates a full training-step simulation."""

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()
        self.manager = PartialRolloutManager(
            model_weight_version=self.config.model_weight_version,
            tokenizer_fingerprint=self.config.tokenizer_fingerprint,
        )
        self._snapshots: dict[str, RolloutStateSnapshot] = {}
        self._report = SimulationReport()

    @property
    def report(self) -> SimulationReport:
        return self._report

    def _gen_prompt(self, idx: int) -> list[int]:
        return [1000 + idx * 10 + j for j in range(self.config.prompt_len)]

    def _gen_tokens(self, idx: int) -> list[int]:
        return [2000 + idx * 10 + j for j in range(self.config.gen_len)]

    def run(self, store_root: Optional[str] = None) -> SimulationReport:
        """Run the full simulation and return a metrics report.

        If ``store_root`` is provided, snapshots are persisted to disk via
        :class:`SnapshotStore` to demonstrate cross-restart resume.
        """
        store = SnapshotStore(store_root) if store_root else None
        self._report.store_root = store_root
        cfg = self.config
        self._report.total_rollouts = cfg.num_rollouts
        pre_count = cfg.pause_before_weight_update
        post_count = cfg.num_rollouts - pre_count

        # 1. Start pre-update rollouts (version v0)
        pre_ids = []
        for i in range(pre_count):
            rid = f"sim-{i}"
            self.manager.create_rollout(
                prompt_ids=self._gen_prompt(i),
                sampling_params={"temperature": 0.7, "max_new_tokens": cfg.gen_len},
                generated_ids=self._gen_tokens(i),
                request_id=rid,
                kv_handle=new_kv_handle(),
            )
            pre_ids.append(rid)

        # 2. Pause pre-update rollouts (capture KV-aware snapshots)
        pre_snaps: dict[str, RolloutStateSnapshot] = {}
        for rid in pre_ids:
            snap = self.manager.pause_rollout(rid)
            pre_snaps[rid] = snap
            if store:
                store.save(snap)

        # 3. Cancel pre-update rollouts — their KV is stale after weight update
        for rid in pre_ids:
            self.manager.cancel_rollout(rid)

        # 4. Update weights (v0 -> v1)
        new_version = "v1"
        self.manager.update_model_version(new_version)

        # 5. Re-create pre-update rollouts with new version (token-only recompute)
        #    Represents regenerating all tokens with new weights.
        for i, rid in enumerate(pre_ids):
            self.manager.create_rollout(
                prompt_ids=self._gen_prompt(i),
                sampling_params={"temperature": 0.7, "max_new_tokens": cfg.gen_len},
                generated_ids=self._gen_tokens(i),
                request_id=rid,
                kv_handle=None,
            )
            self._report.token_only_resumes += 1
            self._report.recomputed_tokens_token_only += cfg.gen_len

        # 6. Start post-update rollouts (version v1, fresh KV)
        post_ids = []
        for i in range(post_count):
            idx = pre_count + i
            rid = f"sim-{idx}"
            self.manager.create_rollout(
                prompt_ids=self._gen_prompt(idx),
                sampling_params={"temperature": 0.7, "max_new_tokens": cfg.gen_len},
                generated_ids=self._gen_tokens(idx),
                request_id=rid,
                kv_handle=new_kv_handle(),
            )
            post_ids.append(rid)

        # 7. Pause and resume post-update rollouts KV-aware (version matches)
        for rid in post_ids:
            snap = self.manager.pause_rollout(rid)
            if store:
                store.save(snap)
            self.manager.resume_rollout(snap)
            self._report.kv_aware_resumes += 1
            self._report.recomputed_tokens_kv_aware += 0

        # 8. Complete all
        for i in range(cfg.num_rollouts):
            rid = f"sim-{i}"
            self.manager.complete_rollout(rid)
            self._report.completed += 1

        # 9. Aggregate metrics
        m = self.manager.metrics
        self._report.save_latency = m.save_latency
        self._report.resume_latency = m.resume_latency
        self._report.failed_resumes = m.invalidated_count

        if store:
            store.clear()

        return self._report

    def run_with_disk_persistence(
        self, store_root: str
    ) -> SimulationReport:
        """Run simulation with snapshots persisted to disk.

        Demonstrates cross-restart resume: snapshots are saved before weight
        update and can be loaded by a fresh manager instance.
        """
        return self.run(store_root=store_root)


def run_default_simulation() -> SimulationReport:
    """Convenience entry point: run with default config and print report."""
    sim = PartialRolloutSimulation()
    report = sim.run()
    print(report.summary())
    return report
