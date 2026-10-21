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
"""Wall-clock diagnostics for weight publication; never introduce device fences."""

import asyncio
import json
import logging
import math
import os
import socket
import time
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
_current_profile: ContextVar = ContextVar("weight_update_profile", default=None)


class WeightUpdateProfile:
    """Record inclusive host wall times and preserve the original failure.

    Nested phases overlap their parents. ``transfer_apply`` includes parameter
    export, transport, and receiver application; it is not wire-only time.
    Begin records remain useful when a remote collective never returns.
    """

    def __init__(self, backend, global_steps, *, operation="weight_update", details=None):
        self.identity = {
            "update_id": uuid4().hex,
            "operation": operation,
            "backend": backend,
            "global_steps": global_steps,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "scope": "host_wall_time",
            "details": dict(details or {}),
        }
        self.durations = {}
        self._sequence = 0
        self.status = "pending"
        self.failed_phase = None
        self.error_type = None

    def _emit(self, phase, status, **fields):
        self._sequence += 1
        record = {
            **self.identity,
            "schema_version": 1,
            "sequence": self._sequence,
            "recorded_at_ns": time.time_ns(),
            "phase": phase,
            "status": status,
            **fields,
        }
        try:
            logger.info("WEIGHT_UPDATE_PROFILE %s", json.dumps(record, sort_keys=True))
        except Exception:
            # Observability must not turn a logging failure into a sync failure.
            pass

    def __enter__(self):
        self._token = _current_profile.set(self)
        self._start = time.perf_counter()
        self.status = "running"
        self._emit("total", "started")
        return self

    def __exit__(self, error_type, error, traceback):
        self.durations["total"] = time.perf_counter() - self._start
        self.status = "completed" if error is None else "failed"
        if isinstance(error, asyncio.CancelledError):
            self.status = "cancelled"
        if error is not None:
            self.error_type = error_type.__name__
        try:
            self._emit(
                "total",
                self.status,
                elapsed_s=self.durations["total"],
                failed_phase=self.failed_phase,
                error_type=self.error_type,
            )
        finally:
            _current_profile.reset(self._token)
        return False

    @contextmanager
    def phase(self, name):
        start = time.perf_counter()
        self._emit(name, "started")
        status = "completed"
        error_type = None
        try:
            yield
        except BaseException as error:
            status = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
            error_type = type(error).__name__
            if self.failed_phase is None:
                self.failed_phase = name
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.durations[name] = self.durations.get(name, 0.0) + elapsed
            self._emit(name, status, elapsed_s=elapsed, error_type=error_type)

    def metrics(self):
        return {f"timing_s/checkpoint/{name}": value for name, value in self.durations.items()}

    def snapshot(self):
        return {
            **self.identity,
            "status": self.status,
            "failed_phase": self.failed_phase,
            "error_type": self.error_type,
            "durations_s": dict(self.durations),
        }


@contextmanager
def weight_update_phase(name):
    """Use the current publication without changing existing method signatures."""
    profile = _current_profile.get()
    if profile is None:
        yield
    else:
        with profile.phase(name):
            yield


def summarize_profile_records(records):
    """Deduplicate copied logs and retain unknown times for unfinished updates."""
    grouped = {}
    for record in records:
        if record.get("schema_version") != 1:
            raise ValueError("Unsupported weight update profile schema")
        sequence = record["sequence"]
        if type(sequence) is not int or sequence < 1:
            raise ValueError("Invalid weight update event sequence")
        elapsed = record.get("elapsed_s")
        if elapsed is not None and (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError("Invalid weight update elapsed time")
        if record["status"] not in {"started", "completed", "failed", "cancelled"}:
            raise ValueError("Invalid weight update event status")
        events = grouped.setdefault(record["update_id"], {})
        if sequence in events and events[sequence] != record:
            raise ValueError("Conflicting copies of a weight update event")
        events[sequence] = record

    profiles = []
    for events in grouped.values():
        ordered = [events[key] for key in sorted(events)]
        identity = ordered[0]
        phases = {}
        durations = {}
        final = None
        first_failure = None
        for row in ordered:
            phase, status = row["phase"], row["status"]
            if status == "started":
                phases[phase] = phases.get(phase, 0) + 1
            else:
                phases[phase] = max(0, phases.get(phase, 0) - 1)
                if row.get("elapsed_s") is not None:
                    durations[phase] = durations.get(phase, 0.0) + row["elapsed_s"]
                if status in {"failed", "cancelled"} and phase != "total" and first_failure is None:
                    first_failure = phase
                if phase == "total":
                    final = row
        profiles.append(
            {
                **{
                    key: identity[key]
                    for key in (
                        "update_id",
                        "operation",
                        "backend",
                        "global_steps",
                        "hostname",
                        "pid",
                        "scope",
                        "details",
                    )
                },
                "status": final["status"] if final else "incomplete",
                "failed_phase": final.get("failed_phase") if final else first_failure,
                "error_type": final.get("error_type") if final else None,
                "durations_s": durations,
                "total_s": durations.get("total"),
                "open_phases": [name for name, count in phases.items() if count],
                "event_count": len(ordered),
            }
        )
    return {
        "schema_version": 1,
        "nested_times_overlap": True,
        "wire_time_s": None,
        "gpu_time_s": None,
        "profiles": profiles,
    }


def main():
    """Read captured driver/worker logs; do not contact Ray or a GPU runtime."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Summarize WEIGHT_UPDATE_PROFILE records from captured logs")
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = []
    decoder = json.JSONDecoder()
    for filename in args.logs:
        with filename.open() as stream:
            for line in stream:
                if "WEIGHT_UPDATE_PROFILE " in line:
                    record, _ = decoder.raw_decode(line.split("WEIGHT_UPDATE_PROFILE ", 1)[1].lstrip())
                    records.append(record)
    if not records:
        parser.error("No weight update profile records found")
    output = json.dumps(summarize_profile_records(records), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()


def weight_component_profile_enabled():
    """Additional boundaries are off until the owner enables milestone 2."""
    return os.getenv("VERL_WEIGHT_COMPONENT_PROFILE", "0") == "1"


def profile_parameter_export(weights, profile):
    """Time original lazy export calls, without consuming tensors early."""
    iterator = iter(weights)
    while True:
        with profile.phase("parameter_export_next"):
            try:
                item = next(iterator)
            except StopIteration:
                return
        yield item


def weight_update_detail(name, **fields):
    """Record the original remote response in the active native profile."""
    profile = _current_profile.get()
    if profile is not None:
        profile._emit(name, "completed", **fields)
