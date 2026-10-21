"""Observe real optimizer calls; preserve native training and checkpoint behavior."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from actor_worker import resolve_training_worker


def _step_frontier(optimizer):
    """Unknown counters stay unknown, including optimizers without step state."""
    values = []
    for state in optimizer.state.values():
        value = state.get("step")
        if hasattr(value, "numel") and value.numel() == 1:
            value = value.item()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values.append(float(value))
    return max(values) if values else None


class OptimizerUpdateAudit:
    """A post-step observer; a call is distinct from verified counter advancement."""

    def __init__(self, optimizer, *, rank, output):
        import torch

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Audit requires the actual torch optimizer")
        if type(rank) is not int or rank < 0:
            raise ValueError("Audit requires the actual nonnegative training rank")
        self.rank = rank
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists():
            raise FileExistsError("Preserve the existing process audit; use a fresh execution prefix")
        self.current_step = None
        self.calls = 0
        self.counter_advancements = 0
        self.initial_frontier = _step_frontier(optimizer)
        self.last_frontier = self.initial_frontier
        self.handle = optimizer.register_step_post_hook(self._after_step)

    def begin_trainer_step(self, step):
        if type(step) is not int or step < 1:
            raise ValueError("Trainer step must be a positive integer")
        if self.current_step is not None and step <= self.current_step:
            raise ValueError("Trainer steps must advance in this process")
        self.current_step = step

    def _after_step(self, optimizer, args, kwargs):
        del args, kwargs
        # Increment first: a receipt I/O failure must never erase a completed update.
        self.calls += 1
        frontier = _step_frontier(optimizer)
        advanced = None if frontier is None else frontier > (self.last_frontier or 0)
        if advanced is True:
            self.counter_advancements += 1
        previous_frontier = self.last_frontier
        self.last_frontier = frontier
        row = self.snapshot()
        row.update(
            optimizer_step_frontier=frontier,
            previous_optimizer_step_frontier=previous_frontier,
            optimizer_counter_advanced=advanced,
        )
        with self.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def snapshot(self):
        return {
            "rank": self.rank,
            "pid": os.getpid(),
            "trainer_step": self.current_step,
            "optimizer_calls_since_attach": self.calls,
            "verified_counter_advancements_since_attach": self.counter_advancements,
            "initial_optimizer_step_frontier": self.initial_frontier,
            "last_optimizer_step_frontier": self.last_frontier,
            "parameter_change_verified": None,
        }

    def close(self):
        self.handle.remove()


def install_worker_update_audit(worker, output_dir):
    """Attach to the original actor optimizer, as in the existing parameter probe."""
    rank = getattr(worker, "rank", None)
    worker = resolve_training_worker(worker)
    if getattr(worker, "_production_update_audit", None) is not None:
        raise RuntimeError("The original actor already has an update audit")
    engine = worker.actor.engine if hasattr(worker, "actor") else worker.engine
    observer = OptimizerUpdateAudit(
        engine.optimizer, rank=rank, output=Path(output_dir) / f"rank-{rank}-pid-{os.getpid()}.jsonl"
    )
    worker._production_update_audit = observer
    return observer.snapshot()


def begin_worker_update_audit_step(worker, step):
    worker = resolve_training_worker(worker)
    worker._production_update_audit.begin_trainer_step(step)
    return worker._production_update_audit.snapshot()


def read_worker_update_audit(worker):
    worker = resolve_training_worker(worker)
    return worker._production_update_audit.snapshot()
