"""Save a stock V1 pre-update batch; replay once through its original actor update.

This module allocates no GPU and starts no rollout server. A replay caller supplies
an already initialized, isolated native trainer and a live identity reader.
Only trusted local DataProto files may be loaded (the native format uses pickle).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import uuid


REQUIRED = {"input_ids", "position_ids", "prompts", "responses", "response_mask",
            "loss_mask", "uid", "old_log_probs", "rm_scores", "advantages", "returns"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _config(trainer):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(trainer.config, resolve=True)


def _training_fingerprint(config):
    # Runtime output paths/topology may change; loss/model/sampling may not.
    scope = {"model": config["actor_rollout_ref"]["model"],
             "actor": config["actor_rollout_ref"]["actor"],
             "algorithm": config["algorithm"],
             "temperature": config["actor_rollout_ref"]["rollout"]["temperature"],
             "n": config["actor_rollout_ref"]["rollout"]["n"]}
    return hashlib.sha256(json.dumps(scope, sort_keys=True, default=str).encode()).hexdigest()


def _identity(value):
    """Same observation shape as run_native_training.publication()."""
    version = value.get("actual_parameter_version")
    record = value.get("active_policy_record", {})
    if type(version) is not int or version < 0:
        raise ValueError("A live observed native parameter version is required")
    if record.get("identity", {}).get("policy_version") != str(version):
        raise ValueError("Live version and published policy record disagree")
    if not record.get("model_configuration_fingerprint"):
        raise ValueError("Published model identity is required")
    return deepcopy(value)


def _rows(value):
    import torch
    if isinstance(value, torch.Tensor):
        return list(value.unbind(0))
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _fields_from_proto(proto):
    from verl.protocol import DataProto
    info = {key: value for key, value in proto.meta_info.items() if key != "_project_b_capture"}
    return DataProto(batch=proto.batch, non_tensor_batch=proto.non_tensor_batch,
                     meta_info=info).to_tensordict()


def validate_native_fields(fields, keys, tags, group_size):
    """Validate native ragged fields without replacing their masks or logprobs."""
    import torch
    missing = REQUIRED - set(fields.keys())
    if missing:
        raise ValueError(f"Missing original pre-update fields: {sorted(missing)}")
    if len(keys) != len(fields) or len(keys) != len(tags) or len(set(keys)) != len(keys):
        raise ValueError("Native keys/tags/rows are not aligned")
    values = {name: _rows(fields[name]) for name in REQUIRED}
    final_sessions = {}
    for row, key in enumerate(keys):
        uid, session, index = key.rsplit("_", 2)
        if str(values["uid"][row]) != uid:
            raise ValueError("Native uid differs from its session key")
        final_sessions[(uid, session)] = max(final_sessions.get((uid, session), -1), int(index))
        prompt, response, ids = (values[name][row] for name in ("prompts", "responses", "input_ids"))
        if not torch.equal(ids, torch.cat((prompt, response))):
            raise ValueError("Native input IDs differ from prompt plus response")
        if len(ids) != tags[row]["seq_len"]:
            raise ValueError("Stored attention length differs from original token length")
        if values["position_ids"][row].shape[-1] != len(ids):
            raise ValueError("Original position IDs are token-misaligned")
        if len(response) != tags[row]["response_len"] or len(prompt) != tags[row]["prompt_len"]:
            raise ValueError("Stored prompt/response lengths differ")
        mask = values["response_mask"][row]
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Original response mask must be finite and binary")
        if len(mask) != len(response):
            raise ValueError("Original response mask has a wrong length")
        # Native rollout correction may reject response tokens after loss_mask
        # was created. Preserve both original fields instead of equating them.
        loss_mask = values["loss_mask"][row]
        if len(loss_mask) != len(response) or not ((loss_mask == 0) | (loss_mask == 1)).all():
            raise ValueError("Original loss mask is not a response-aligned binary mask")
        for name in ("old_log_probs", "rm_scores", "advantages", "returns"):
            value = values[name][row]
            if len(value) != len(response) or not torch.isfinite(value).all():
                raise ValueError(f"Original {name} is nonfinite or response-misaligned")
    groups = {}
    for uid, session in final_sessions:
        groups.setdefault(uid, set()).add(session)
    if not groups or any(len(members) != group_size for members in groups.values()):
        raise ValueError("Replay requires complete native GRPO prompt groups")
    return {"rows": len(keys), "prompt_groups": len(groups), "members_per_group": group_size,
            "fields": sorted(str(name) for name in fields.keys()),
            "attention_representation": "native_unpadded_input_ids_and_seq_len; all original tokens attend"}


def capture_before_update(trainer, batch, destination, *, read_identity):
    """Capture all original TransferQueue fields immediately before _update_actor."""
    import transfer_queue as tq
    from verl.protocol import DataProto

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    config = _config(trainer)
    identity = _identity(read_identity(trainer))
    # No select_fields: retain optional KL, rollout correction, multimodal and MoE fields.
    fields = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id).cpu()
    keys, tags = list(batch.keys), deepcopy(batch.tags)
    summary = validate_native_fields(fields, keys, tags, config["actor_rollout_ref"]["rollout"]["n"])
    if summary["prompt_groups"] != config["data"]["train_batch_size"]:
        raise ValueError("Native update batch must match the configured complete prompt-group count")
    metadata = {"keys": keys, "tags": tags, "extra_info": deepcopy(batch.extra_info),
                "partition_id": batch.partition_id, "fields": deepcopy(batch.fields)}
    # Reuse the project's existing native serializer; do not pad/re-tokenize.
    proto = DataProto.from_tensordict(fields)
    proto.meta_info["_project_b_capture"] = metadata
    data_path = destination / "native-batch.pkl"
    proto.save_to_disk(data_path)
    _json_write(destination / "config.json", config)
    manifest = {"schema": "project-b-native-pre-update-v1", "data_kind": "real_native_rollout",
                "stage": "after_original_advantage_before_actor_update",
                "trainer_step": trainer.global_steps, "identity": identity,
                "training_fingerprint": _training_fingerprint(config),
                "batch_sha256": _sha(data_path), "config_sha256": _sha(destination / "config.json"),
                "summary": summary, "optimizer_executed_by_capture": False,
                "online_training_closed_loop_verified": False}
    _json_write(destination / "manifest.json", manifest)
    signal = getattr(trainer, "audit_learning_signals", {}).get(trainer.global_steps)
    if isinstance(signal, dict) and signal.get("nonzero_training_advantage") is False:
        _json_write(destination / "no-grpo-learning-signal.json", {
            "status": "NO_GRPO_LEARNING_SIGNAL_BEFORE_ACTOR_UPDATE",
            "trainer_step": trainer.global_steps,
            "actor_update_dispatched": False,
            "original_learning_signal": deepcopy(signal),
            "capture_manifest": manifest,
        })
        raise RuntimeError("NO_GRPO_LEARNING_SIGNAL: saved the real native batch; actor update was not dispatched")
    return manifest


def prepare_real_rollout_capture(trainer, batch, destination, *, read_identity):
    """Compute missing training fields with original model methods, then save.

    The caller supplies a real, rewarded native rollout KVBatchMeta. This runs
    original inference/advantage stages on the caller's training allocation and
    performs no optimizer call. It never requests rollout generation.
    """
    import transfer_queue as tq
    correction = trainer.config.algorithm.get("rollout_correction", None)
    if correction and correction.get("bypass_mode", False):
        raise ValueError("This diagnostic requires native actor recomputation of old_log_probs; bypass mode is not allowed")
    raw = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id)
    if "rm_scores" not in raw.keys():
        raise ValueError("A real reward must be supplied; this entry does not fabricate rewards")
    metrics = {}
    batch = trainer._balance_batch(batch, metrics=metrics)
    batch = trainer._compute_old_log_prob(batch, metrics=metrics)
    if trainer.use_reference_policy:
        batch = trainer._compute_ref_log_prob(batch, metrics=metrics)
    if trainer.use_critic:
        batch = trainer._compute_values(batch, metrics=metrics)
    batch = trainer._compute_advantage(batch, metrics=metrics)
    return capture_before_update(trainer, batch, destination, read_identity=read_identity), metrics


def inspect_capture(directory):
    """Validate a trusted native capture; never initialize a model or optimizer."""
    from verl.protocol import DataProto
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "project-b-native-pre-update-v1":
        raise ValueError("Unsupported capture schema")
    if _sha(root / "native-batch.pkl") != manifest["batch_sha256"]:
        raise ValueError("Native batch bytes changed")
    if _sha(root / "config.json") != manifest["config_sha256"]:
        raise ValueError("Captured config bytes changed")
    proto = DataProto.load_from_disk(root / "native-batch.pkl")
    _identity(manifest["identity"])
    config = json.loads((root / "config.json").read_text())
    info = proto.meta_info["_project_b_capture"]
    summary = validate_native_fields(_fields_from_proto(proto), info["keys"], info["tags"],
                                     config["actor_rollout_ref"]["rollout"]["n"])
    if summary != manifest["summary"]:
        raise ValueError("Native capture metadata differs after roundtrip")
    return proto, manifest


def replay_one_update(trainer, directory, *, read_identity, result_path):
    """Dispatch once through the existing trainer method; never publish a version.

    trainer must already own isolated initialized original actor workers. Native
    _update_actor -> actor_rollout_wg.update_actor -> train_mini_batch performs
    the original forward, backward and optimizer. No rollout is requested here.
    """
    import transfer_queue as tq
    proto, manifest = inspect_capture(directory)
    live = _identity(read_identity(trainer))
    if live != manifest["identity"]:
        raise ValueError("Replay actor must load the exact captured policy identity")
    if _training_fingerprint(_config(trainer)) != manifest["training_fingerprint"]:
        raise ValueError("Replay model/loss/sampling config differs from capture")
    if trainer.global_steps != manifest["trainer_step"]:
        raise ValueError("Replay trainer step differs from capture")
    # A new partition preserves real keys and prevents overwriting live train data.
    info = proto.meta_info["_project_b_capture"]
    batch = tq.kv_batch_put(keys=info["keys"], partition_id="offline-replay-" + uuid.uuid4().hex,
                            tags=info["tags"], fields=_fields_from_proto(proto))
    batch.extra_info.update(deepcopy(info["extra_info"]))
    result_path = Path(result_path)
    # Reserve the result before dispatch; an ambiguous failure is never auto-retried.
    _json_write(result_path, {"status": "DISPATCHED_REQUIRES_READBACK", "capture": str(Path(directory).resolve()),
                             "online_training_closed_loop_verified": False})
    metrics = {}
    trainer._update_actor(batch, metrics=metrics)
    result = {"status": "ORIGINAL_UPDATE_RETURNED_NOT_YET_PARAMETER_PROOF", "metrics": metrics,
              "capture_sha256": manifest["batch_sha256"],
              "optimizer_and_parameter_delta_independently_verified": False,
              "online_training_closed_loop_verified": False}
    # Preserve native metrics, with no substituted gradient/delta/version values.
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False, default=str) + "\n")
    return result


def capture_trainer_class(base, output_root, *, read_identity):
    """Wrap a chosen trainer before construction; the entire stock update remains."""
    class CaptureTrainer(base):
        def _update_actor(self, batch, metrics):
            root = Path(output_root) / f"pre-update-{self.global_steps}"
            capture_before_update(self, batch, root, read_identity=read_identity)
            return super()._update_actor(batch, metrics)
    return CaptureTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--trainer-factory", help="module:function returning (initialized_trainer, live_identity_reader)")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.trainer_factory:
        if args.result is None:
            parser.error("--result is required for one-shot replay")
        module, function = args.trainer_factory.split(":", 1)
        trainer, reader = getattr(importlib.import_module(module), function)()
        print(json.dumps(replay_one_update(trainer, args.capture, read_identity=reader,
                                           result_path=args.result), default=str))
    else:
        _, manifest = inspect_capture(args.capture)
        print(json.dumps({"status": "CAPTURE_VALIDATED_NO_TRAINING", **manifest}, indent=2))


if __name__ == "__main__":
    main()
