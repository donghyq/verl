"""Budget and native-save receipts; never serialize model/optimizer tensors here."""
from __future__ import annotations

import functools
import contextlib
import contextvars
import hashlib
import json
import math
import os
import socket
import time
from pathlib import Path

MARKER = "NATIVE_CHECKPOINT_COMPLETE.json"
_ACTIVE_NATIVE_SAVE = contextvars.ContextVar("native_checkpoint_save_receipt", default=None)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def append_event(path, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(dict(unix_s=time.time(), monotonic_s=time.monotonic(),
                                     pid=os.getpid(), **fields), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def observe_call(path, stage, **fields):
    started = time.monotonic()
    append_event(path, stage=stage, status="BEGIN", **fields)
    try:
        yield
    except BaseException as error:
        append_event(path, stage=stage, status="EXCEPTION", error_type=type(error).__name__,
                     elapsed_s=time.monotonic()-started, **fields)
        raise
    else:
        append_event(path, stage=stage, status="RETURNED", elapsed_s=time.monotonic()-started, **fields)


@contextlib.contextmanager
def observe_bound_method(obj, name, path, stage, **fields):
    original = getattr(obj, name)
    had_own_attribute = name in getattr(obj, "__dict__", {})
    @functools.wraps(original)
    def observed(*args, **kwargs):
        with observe_call(path, stage, **fields):
            return original(*args, **kwargs)
    setattr(obj, name, observed)
    try:
        yield
    finally:
        if had_own_attribute:
            setattr(obj, name, original)
        else:
            delattr(obj, name)


class _CheckpointTorchProxy:
    """Forward the original module; observe only this save call's torch.save."""
    def __init__(self, original): self.original = original
    def __getattr__(self, name): return getattr(self.original, name)
    def save(self, value, destination, *args, **kwargs):
        active = _ACTIVE_NATIVE_SAVE.get()
        if active is None:
            return self.original.save(value, destination, *args, **kwargs)
        log, fields = active
        with observe_call(log, "native_torch_save", destination=str(destination), **fields):
            result = self.original.save(value, destination, *args, **kwargs)
        path = Path(destination)
        append_event(log, stage="native_file_size_after_save", status="OBSERVED",
                     destination=str(path), bytes=path.stat().st_size if path.is_file() else None,
                     native_write_return_is_not_physical_storage_durability_proof=True, **fields)
        return result


def require_save_budget(options, *, now=None):
    now = time.time() if now is None else now
    keys = ("work_deadline_unix_s", "save_reserve_s", "post_save_reserve_s")
    if not isinstance(options, dict) or any(type(options.get(k)) not in (int, float)
            or not math.isfinite(options[k]) for k in keys):
        raise ValueError("Explicit finite native checkpoint budget is required")
    deadline, reserve, post = (options[k] for k in keys)
    if reserve <= 0 or post < 0 or deadline - now < reserve + post:
        raise TimeoutError("NATIVE_CHECKPOINT_SAVE_BUDGET_INSUFFICIENT")
    return dict(remaining_work_s=deadline-now, save_reserve_s=reserve,
                post_save_reserve_s=post, deadline_is_not_extended=True)


def inventory(root, step, *, rank_count=4):
    """Check small metadata and native file sizes; no pickle and no full-weight SHA."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or root.name != f"global_step_{step}":
        raise ValueError("Native checkpoint root/frontier is missing")
    required = [f"actor/{prefix}_world_size_{rank_count}_rank_{rank}.pt"
                for rank in range(rank_count) for prefix in ("model", "optim", "extra_state")]
    required += ["actor/fsdp_config.json", "data.pt", "trajectory_state/manifest.json"]
    files = []
    for relative in required:
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError("Incomplete native checkpoint: " + relative)
        files.append(dict(path=relative, bytes=path.stat().st_size))
    queue = root / "transfer_queue"
    if queue.is_symlink() or not queue.is_dir():
        raise ValueError("Native TransferQueue checkpoint is missing")
    queue_files = [p for p in queue.rglob("*") if p.is_file()]
    if not queue_files or any(p.is_symlink() or p.stat().st_size <= 0 for p in queue_files):
        raise ValueError("Native TransferQueue checkpoint is empty or invalid")
    files += [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size) for p in sorted(queue_files)]
    trajectory = json.loads((root / "trajectory_state/manifest.json").read_text())
    if (trajectory.get("checkpoint_step") != step or trajectory.get("quiesced") is not True
            or int(trajectory["last_published_version"]) >= step):
        raise ValueError("Native trajectory checkpoint frontier differs")
    for relative, digest in trajectory["files"].items():
        name = Path(relative)
        path = root / "trajectory_state" / name
        if name.is_absolute() or ".." in name.parts or path.is_symlink() or not path.is_file():
            raise ValueError("Native trajectory reference escapes checkpoint")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Native trajectory reference digest differs")
    latest = root.parent / "latest_checkpointed_iteration.txt"
    if latest.is_symlink() or not latest.is_file() or latest.read_text().strip() != str(step):
        raise ValueError("Native latest iteration has not committed this step")
    return dict(step=step, world_size=rank_count, files=files,
                trajectory_manifest_sha256=hashlib.sha256((root/"trajectory_state/manifest.json").read_bytes()).hexdigest(),
                full_weight_sha_performed=False, checkpoint_tensors_loaded_by_guard=False)


def install_worker_save_observer(worker, directory):
    """Wrap the actual bound engine method once; collect original returns/errors."""
    import torch
    import ray
    from actor_worker import resolve_training_worker

    selected = resolve_training_worker(worker)
    engine = selected.actor.engine if hasattr(selected, "actor") else selected.engine
    rank = torch.distributed.get_rank()
    context = ray.get_runtime_context()
    identity = dict(hostname=socket.gethostname(),ray_actor_id=str(context.get_actor_id()),
                    ray_node_id=str(context.get_node_id()))
    if getattr(engine, "_native_checkpoint_save_observer", False):
        raise ValueError("Native checkpoint observer already installed")
    original = engine.save_checkpoint
    path = Path(directory) / f"rank-{rank}.jsonl"

    @functools.wraps(original)
    def observed(*args, **kwargs):
        local_path = str(kwargs.get("local_path", args[0] if args else ""))
        global_step = kwargs.get("global_step", args[2] if len(args) > 2 else None)
        fields = dict(rank=rank, local_path=local_path, global_step=global_step, **identity)
        started = time.monotonic()
        append_event(path, status="ORIGINAL_NATIVE_SAVE_BEGIN", **fields)
        try:
            # The native implementation still creates the state dicts and calls
            # torch.save. The temporary wrappers forward each original call once.
            manager = engine.checkpoint_manager
            namespace = manager.save_checkpoint.__func__.__globals__
            original_torch = namespace["torch"]
            token = _ACTIVE_NATIVE_SAVE.set((path, fields))
            namespace["torch"] = _CheckpointTorchProxy(original_torch)
            try:
                with contextlib.ExitStack() as stack:
                    for obj, stage in ((manager.model, "model_state_dict"),
                                       (manager.optimizer, "optimizer_state_dict"),
                                       (manager.lr_scheduler, "scheduler_state_dict")):
                        if obj is not None:
                            stack.enter_context(observe_bound_method(obj, "state_dict", path, stage, **fields))
                    result = original(*args, **kwargs)
            finally:
                namespace["torch"] = original_torch
                _ACTIVE_NATIVE_SAVE.reset(token)
        except BaseException as error:
            append_event(path, status="ORIGINAL_NATIVE_SAVE_EXCEPTION", error_type=type(error).__name__,
                         elapsed_s=time.monotonic()-started, **fields)
            raise
        append_event(path, status="ORIGINAL_NATIVE_SAVE_RETURNED", elapsed_s=time.monotonic()-started, **fields)
        return result

    engine.save_checkpoint = observed
    engine._native_checkpoint_save_observer = True
    return dict(rank=rank, pid=os.getpid(), status="INSTALLED_ON_ORIGINAL_ENGINE_METHOD")


def complete_native_save(root, step, *, receipt_directory, elapsed_s):
    layout = inventory(root, step)
    returns = []
    for rank in range(4):
        path = Path(receipt_directory) / f"rank-{rank}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        matches = [r for r in rows if r.get("status") == "ORIGINAL_NATIVE_SAVE_RETURNED"
                   and r.get("rank") == rank and r.get("global_step") == step
                   and Path(r["local_path"]).resolve() == (Path(root)/"actor").resolve()]
        if len(matches) != 1:
            raise ValueError("Missing or repeated original native save return for rank " + str(rank))
        returns.append(matches[0])
    value = dict(status="ORIGINAL_NATIVE_SAVE_RETURNED_AND_LAYOUT_COMPLETE", **layout,
                 native_rank_returns=returns, driver_elapsed_s=elapsed_s,
                 actual_restore_verified=False, model_optimizer_contents_verified_by_native_load=False)
    atomic_json(Path(root)/MARKER, value)
    return value


def require_complete_checkpoint(root, step):
    marker = Path(root)/MARKER
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("NATIVE_CHECKPOINT_COMPLETE marker missing; trajectory state alone is not a training checkpoint")
    value = json.loads(marker.read_text())
    actual = inventory(root, step)
    if (value.get("status") != "ORIGINAL_NATIVE_SAVE_RETURNED_AND_LAYOUT_COMPLETE"
            or value.get("step") != step or value.get("world_size") != 4
            or value.get("files") != actual["files"]
            or value.get("trajectory_manifest_sha256") != actual["trajectory_manifest_sha256"]
            or {r.get("rank") for r in value.get("native_rank_returns", [])} != set(range(4))):
        raise ValueError("Native checkpoint completion metadata differs")
    return value
