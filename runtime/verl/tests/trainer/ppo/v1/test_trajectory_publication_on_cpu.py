# Copyright 2026 Project B contributors
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

"""CPU protocol mocks and one opt-in, isolated real TransferQueue smoke.

No test loads a model or executes a training optimizer. Mock observations are
not business evidence. The real TQ smoke proves only public data readback.
"""

import asyncio
import hashlib
import os
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

from verl.trainer.ppo.v1.trajectory_publication import (
    PublicationMember,
    publish_trajectory_group,
    reconcile_trajectory_group_publication,
    training_payload_digest,
)
from verl.trainer.ppo.v1.trajectory_state import PublicationBlocked
from verl.trainer.ppo.v1.trajectory_state.handoff import HandoffState
from verl.utils.tensordict_utils import get as td_get
from verl.utils.tensordict_utils import list_of_dict_to_tensordict


UID = "fixture-group"
MEMBERS = tuple(PublicationMember(UID, i, 0) for i in range(8))


def original_fields():
    rows = []
    for i in range(8):
        prompt = torch.arange(101, 103 + i % 3, dtype=torch.int64)
        response = torch.tensor([11] * (2 + i % 4), dtype=torch.int64)
        mask = torch.ones_like(response)
        rows.append({
            "uid": UID, "session_id": i, "prompts": prompt, "responses": response,
            "response_mask": mask, "loss_mask": mask.clone(),
            "rollout_log_probs": torch.full(response.shape, -0.2, dtype=torch.float32),
            "input_ids": torch.cat((prompt, response)),
            "position_ids": torch.arange(prompt.numel() + response.numel(), dtype=torch.int64),
            "rm_scores": torch.zeros(response.shape, dtype=torch.float32),
            "extra_fields": {"policy_version": "fixture-v0", "execution_generation": 1,
                             "trajectory_segments": [{"policy_version": "fixture-v0", "start": 0, "end": len(response)}]},
        })
    return list_of_dict_to_tensordict(rows)


def original_tags():
    return [{"status": "success", "global_steps": 0, "min_global_steps": 0, "max_global_steps": 0} for _ in range(8)]


def original_validator(fields, members):
    assert len(members) == 8
    for row in fields.unbind(0):
        if td_get(row, "extra_fields")["policy_version"] != "fixture-v0":
            raise ValueError("Fixture original policy validator rejected the version")


class FakeCoordinator:
    def __init__(self):
        self.publication = None
        self.manifest_state = HandoffState.RESTORING.value
        self.calls = []
        self.confirm_failures = 0
        self.confirm_count = 0
        self.confirm_fail_on = None
        self.confirm_response_fail_on = None
        self.finish_failures = 0
        self.gc_fails = False
        self.uid = UID

    def get_handoff_manifest(self, handoff_id):
        return SimpleNamespace(member_tables=lambda: {self.uid: {"group_size": 8}})

    def get_publication(self, handoff_id):
        return deepcopy(self.publication)

    def summary(self):
        return {"handoffs": {"fixture-handoff": self.manifest_state}}

    def prepare_publication(self, handoff_id, binding):
        self.calls.append("prepare")
        assert self.manifest_state == HandoffState.RESTORING.value
        self.publication = {
            "publication_id": hashlib.sha256(b"fixture publication").hexdigest(),
            "binding": deepcopy(binding), "state": "PREPARED", "receipt": None,
        }
        return deepcopy(self.publication)

    def mark_publication_started(self, handoff_id):
        self.calls.append("mark_started")
        if self.publication["state"] != "PREPARED":
            raise PublicationBlocked("No retry after a dispatch")
        self.publication["state"] = "DISPATCHED"
        return deepcopy(self.publication)

    def confirm_publication(self, handoff_id, receipt):
        self.calls.append("confirm")
        self.confirm_count += 1
        if self.confirm_failures > 0:
            self.confirm_failures -= 1
            raise OSError("Fixture metadata response failed after actual GET")
        if self.confirm_count == self.confirm_fail_on:
            raise OSError("Fixture metadata failed before ready proof was committed")
        assert receipt["binding"] == self.publication["binding"]
        assert receipt["backend_reference"]["verified_members"] == 8
        self.publication.update(state="PUBLISHED", receipt=deepcopy(receipt))
        if self.confirm_count == self.confirm_response_fail_on:
            raise OSError("Fixture metadata committed but its response was lost")
        return deepcopy(self.publication)

    def reconcile_publication(self, handoff_id, receipt):
        self.calls.append("reconcile")
        assert receipt is None
        if self.publication["state"] != "PREPARED":
            self.publication["state"] = "BLOCKED"
        return deepcopy(self.publication)

    def finish_handoff_manifest(self, handoff_id, publication_id):
        self.calls.append("finish")
        if self.finish_failures:
            self.finish_failures -= 1
            raise OSError("Fixture finish metadata failed")
        assert publication_id == self.publication["publication_id"]
        assert self.publication["state"] == "PUBLISHED"
        assert self.publication["receipt"]["backend_reference"]["prompt_ready"]["verified"] is True
        self.manifest_state = HandoffState.FINISHED.value
        return self.manifest_state

    def collect_garbage(self):
        self.calls.append("gc")
        assert self.manifest_state == HandoffState.FINISHED.value
        if self.gc_fails:
            raise OSError("Fixture cleanup failure")
        return 3


class FakeQueue:
    def __init__(self, coordinator, *, put_mode="normal", mutation=None):
        self.coordinator = coordinator
        self.put_mode = put_mode
        self.mutation = mutation
        self.fields = None
        self.tags = None
        self.keys = None
        self.requested_keys = None
        self.partition_id = None
        self.put_calls = 0
        self.get_calls = 0
        self.list_calls = 0
        self.prompt_tag = {"status": "running"}
        self.prompt_put_calls = 0
        self.prompt_put_mode = "normal"
        self.prompt_read_failure_after_put = False

    async def async_kv_batch_put(self, *, keys, fields, tags, partition_id):
        self.coordinator.calls.append("put")
        assert self.coordinator.publication["state"] == "DISPATCHED"
        self.put_calls += 1
        self.requested_keys, self.partition_id = deepcopy(keys), partition_id
        if self.put_mode == "partial":
            # TensorDict range slicing can raise on jagged tensors. Do not first
            # retain the complete batch and accidentally model a lost response
            # to a successful full PUT when that slicing operation fails.
            self.keys, self.tags = deepcopy(keys[:4]), deepcopy(tags[:4])
            self.fields = None
            rows = [{name: td_get(row, name) for name in fields.keys()}
                    for row in fields.unbind(0)[:4]]
            self.fields = list_of_dict_to_tensordict(rows)
        else:
            self.keys = deepcopy(keys)
            self.fields, self.tags = deepcopy(fields), deepcopy(tags)
        if self.put_mode == "metadata_lost":
            self.tags = None
            raise OSError("Fixture data PUT succeeded but queue metadata failed")
        if self.put_mode == "return_failed":
            raise OSError("Fixture PUT response lost after data and metadata persisted")
        return SimpleNamespace(status="success", keys=keys)

    async def async_kv_batch_get(self, *, keys, partition_id, select_fields=None):
        self.coordinator.calls.append("get")
        self.get_calls += 1
        assert select_fields is None
        if self.fields is None or self.tags is None:
            raise ValueError("Fixture missing/unready queue data")
        assert keys == self.requested_keys and partition_id == self.partition_id
        actual = deepcopy(self.fields)
        if self.mutation:
            actual = self.mutation(actual)
        return actual

    async def async_kv_list(self, *, partition_id):
        self.coordinator.calls.append("list")
        self.list_calls += 1
        if self.prompt_read_failure_after_put and self.prompt_put_calls:
            raise OSError("Fixture prompt ready metadata read failed")
        values = {UID: deepcopy(self.prompt_tag)} if self.prompt_tag is not None else {}
        if self.tags is None and self.put_calls:
            raise ValueError("Fixture missing queue metadata")
        if self.tags is not None:
            values.update(dict(zip(self.keys, deepcopy(self.tags), strict=True)))
        return {partition_id: values}

    async def async_kv_put(self, *, key, partition_id, tag):
        self.coordinator.calls.append("put_uid")
        self.prompt_put_calls += 1
        assert key == UID and self.prompt_tag["status"] == "running"
        assert self.coordinator.publication["state"] == "PUBLISHED"
        assert self.coordinator.publication["receipt"]["backend_reference"]["verified_members"] == 8
        if self.prompt_put_mode == "not_applied":
            raise OSError("Fixture ready write was not applied")
        self.prompt_tag.update(deepcopy(tag))
        if self.prompt_put_mode == "response_failed":
            raise OSError("Fixture ready write completed but its metadata response failed")


def run_publish(coordinator, queue, **overrides):
    params = dict(
        coordinator=coordinator, handoff_id="fixture-handoff", members=MEMBERS,
        fields=original_fields(), tags=original_tags(), partition_id="train",
        validate_fields=original_validator, tq_api=queue,
    )
    params.update(overrides)
    return asyncio.run(publish_trajectory_group(**params))


def test_real_payload_is_verified_and_finish_precedes_gc():
    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator)
    result = run_publish(coordinator, queue)
    assert result["state"] == "FINISHED"
    assert result["readback_verified"]
    assert result["gc_count"] == 3
    assert coordinator.calls == ["list", "prepare", "mark_started", "put", "get", "list", "confirm",
                                 "list", "put_uid", "list", "confirm", "finish", "gc"]
    assert result["prompt_ready_verified"]
    assert result["receipt"]["backend_reference"]["prompt_ready"]["actual_status"] == "finished"
    assert result["receipt"]["backend_reference"]["proof_layer"] == "injected_api_only"
    assert not result["queue_replay_persistence_proved"]
    assert not result["optimizer_ack_atomicity_proved"]


def test_dictionary_order_and_tensor_storage_do_not_change_digest():
    fields = original_fields()
    permuted = fields.select(*reversed(list(fields.keys())))
    keys = [member.key for member in MEMBERS]
    first = training_payload_digest(keys=keys, partition_id="train", fields=fields, tags=original_tags())
    second = training_payload_digest(keys=keys, partition_id="train", fields=deepcopy(permuted), tags=original_tags())
    assert first == second


def test_partition_and_original_key_order_are_bound_to_digest():
    fields, keys = original_fields(), [member.key for member in MEMBERS]
    baseline = training_payload_digest(keys=keys, partition_id="train", fields=fields, tags=original_tags())
    assert training_payload_digest(keys=keys, partition_id="val", fields=fields, tags=original_tags()) != baseline
    assert training_payload_digest(keys=list(reversed(keys)), partition_id="train", fields=fields, tags=original_tags()) != baseline


def test_partial_put_return_code_cannot_confirm_a_group():
    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator, put_mode="partial")
    result = run_publish(coordinator, queue)
    assert result["state"] == "BLOCKED"
    assert result["reason"] == "ACTUAL_READBACK_FAILED"
    assert "write_error_type" not in result  # The partial fixture itself must not fail.
    assert list(queue.fields.batch_size) == [4]
    assert queue.keys == [member.key for member in MEMBERS[:4]] and len(queue.tags) == 4
    assert not result["readback_verified"]
    assert coordinator.publication["state"] == "BLOCKED"
    assert "confirm" not in coordinator.calls and "gc" not in coordinator.calls
    again = run_publish(coordinator, queue)
    assert again["state"] == "BLOCKED"
    assert queue.put_calls == 1


def test_write_with_lost_queue_metadata_is_blocked_without_resending():
    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator, put_mode="metadata_lost")
    assert run_publish(coordinator, queue)["state"] == "BLOCKED"
    assert queue.fields is not None
    assert run_publish(coordinator, queue)["state"] == "BLOCKED"
    assert queue.put_calls == 1
    assert "gc" not in coordinator.calls


def test_actual_readback_can_prove_a_write_whose_response_was_lost():
    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator, put_mode="return_failed")
    result = run_publish(coordinator, queue)
    assert result["state"] == "FINISHED"
    assert result["readback_verified"]
    assert result["write_error_type"] == "OSError"
    assert queue.put_calls == 1


def test_behavior_logprob_tampering_cannot_be_confirmed():
    def tamper(fields):
        values = list(fields.get("rollout_log_probs").unbind(0))
        values[0] = values[0].clone()
        values[0][0] -= 0.1
        fields.set("rollout_log_probs", torch.nested.as_nested_tensor(values, layout=torch.jagged))
        return fields

    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator, mutation=tamper)
    result = run_publish(coordinator, queue)
    assert result["state"] == "BLOCKED"
    assert "confirm" not in coordinator.calls


def test_missing_actual_training_column_is_blocked():
    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator, mutation=lambda fields: fields.exclude("rollout_log_probs"))
    assert run_publish(coordinator, queue)["state"] == "BLOCKED"
    assert "confirm" not in coordinator.calls


def test_policy_validator_runs_on_actual_readback():
    calls = []

    def policy_check(fields, members):
        calls.append("validate")
        if len(calls) == 2:
            raise ValueError("Fixture policy mismatch in actual data")
        original_validator(fields, members)

    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator)
    assert run_publish(coordinator, queue, validate_fields=policy_check)["state"] == "BLOCKED"
    assert calls == ["validate", "validate"]
    assert "confirm" not in coordinator.calls


def test_foreign_handoff_group_is_rejected_before_queue_io():
    coordinator = FakeCoordinator()
    coordinator.uid = "another-group"
    queue = FakeQueue(coordinator)
    result = run_publish(coordinator, queue)
    assert result["state"] == "BLOCKED"
    assert queue.put_calls == 0 and queue.get_calls == 0


def test_confirmation_metadata_failure_preserves_fields_and_allows_read_only_recovery():
    coordinator = FakeCoordinator()
    coordinator.confirm_failures = 1
    queue = FakeQueue(coordinator)
    first = run_publish(coordinator, queue)
    assert first["state"] == "BLOCKED" and first["readback_verified"]
    assert "gc" not in coordinator.calls
    second = run_publish(coordinator, queue)
    assert second["state"] == "FINISHED" and second["readback_verified"]
    assert not second["write_attempted"]
    assert queue.put_calls == 1 and queue.get_calls == 2


def test_published_but_unfinished_recovery_reuses_durable_ready_proof():
    coordinator = FakeCoordinator()
    coordinator.finish_failures = 1
    queue = FakeQueue(coordinator)
    assert run_publish(coordinator, queue)["state"] == "BLOCKED"
    coordinator.publication["state"] = "NEEDS_VERIFICATION"
    assert run_publish(coordinator, queue)["state"] == "FINISHED"
    assert queue.put_calls == 1 and queue.get_calls == 1
    assert queue.prompt_put_calls == 1


def test_missing_data_after_dispatch_is_not_recreated():
    coordinator = FakeCoordinator()
    coordinator.confirm_failures = 1
    queue = FakeQueue(coordinator)
    assert run_publish(coordinator, queue)["state"] == "BLOCKED"
    queue.fields = None
    assert run_publish(coordinator, queue)["state"] == "BLOCKED"
    assert queue.put_calls == 1
    assert coordinator.publication["state"] == "BLOCKED"


def test_cleanup_failure_does_not_undo_durable_finished_delivery():
    coordinator = FakeCoordinator()
    coordinator.gc_fails = True
    queue = FakeQueue(coordinator)
    result = run_publish(coordinator, queue)
    assert result["state"] == "FINISHED"
    assert result["gc_count"] is None and result["gc_error_type"] == "OSError"
    assert coordinator.manifest_state == HandoffState.FINISHED.value


async def async_validator(fields, members):
    original_validator(fields, members)
    return True


def test_async_original_validator_is_supported():
    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator)
    assert run_publish(coordinator, queue, validate_fields=async_validator)["state"] == "FINISHED"


def test_duplicate_member_keys_rejected_before_intent():
    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator)
    duplicate = [*MEMBERS[:-1], MEMBERS[0]]
    result = run_publish(coordinator, queue, members=duplicate)
    assert result["state"] == "BLOCKED"
    assert coordinator.publication is None and queue.put_calls == 0


def test_changed_fields_cannot_rewrite_a_dispatched_binding():
    coordinator = FakeCoordinator()
    coordinator.confirm_failures = 1
    queue = FakeQueue(coordinator)
    assert run_publish(coordinator, queue)["state"] == "BLOCKED"
    changed_tags = original_tags()
    changed_tags[0]["global_steps"] = 1
    assert run_publish(coordinator, queue, tags=changed_tags)["state"] == "BLOCKED"
    assert queue.put_calls == 1


def recover(coordinator, queue, validator=original_validator):
    return asyncio.run(reconcile_trajectory_group_publication(
        coordinator=coordinator, handoff_id="fixture-handoff", validate_fields=validator, tq_api=queue,
    ))


def test_original_uid_must_exist_as_running_before_first_member_write():
    for status in (None, "finished", "consumed", "unknown"):
        coordinator = FakeCoordinator()
        queue = FakeQueue(coordinator)
        queue.prompt_tag = None if status is None else {"status": status}
        result = run_publish(coordinator, queue)
        assert result["state"] == "BLOCKED"
        assert queue.put_calls == 0 and queue.prompt_put_calls == 0
        assert "gc" not in coordinator.calls


def test_ready_put_response_failure_requires_actual_finished_readback():
    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator)
    queue.prompt_put_mode = "response_failed"
    result = run_publish(coordinator, queue)
    assert result["state"] == "FINISHED" and result["prompt_ready_verified"]
    assert result["prompt_ready_write_error_type"] == "OSError"
    assert queue.prompt_put_calls == 1


def test_unobserved_ready_never_allows_finish_or_gc():
    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator)
    queue.prompt_put_mode = "not_applied"
    result = run_publish(coordinator, queue)
    assert result["state"] == "BLOCKED"
    assert not result["prompt_ready_verified"]
    assert "finish" not in coordinator.calls and "gc" not in coordinator.calls
    assert queue.prompt_tag["status"] == "running"


def test_ready_write_followed_by_metadata_failure_recovers_without_rewriting_uid():
    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator)
    queue.prompt_read_failure_after_put = True
    first = run_publish(coordinator, queue)
    assert first["state"] == "BLOCKED"
    assert queue.prompt_tag["status"] == "finished"
    assert "finish" not in coordinator.calls and "gc" not in coordinator.calls
    queue.prompt_read_failure_after_put = False
    second = recover(coordinator, queue)
    assert second["state"] == "FINISHED"
    assert queue.put_calls == 1 and queue.prompt_put_calls == 1


def test_ready_confirmation_failure_does_not_delete_snapshots():
    coordinator = FakeCoordinator()
    coordinator.confirm_fail_on = 2
    queue = FakeQueue(coordinator)
    first = run_publish(coordinator, queue)
    assert first["state"] == "BLOCKED" and first["prompt_ready_verified"]
    assert "prompt_ready" not in coordinator.publication["receipt"]["backend_reference"]
    assert "finish" not in coordinator.calls and "gc" not in coordinator.calls
    second = recover(coordinator, queue)
    assert second["state"] == "FINISHED"
    assert queue.put_calls == 1 and queue.prompt_put_calls == 1


def test_no_ready_proof_and_consumed_payload_stays_blocked_without_retraining():
    coordinator = FakeCoordinator()
    coordinator.confirm_fail_on = 2
    queue = FakeQueue(coordinator)
    assert run_publish(coordinator, queue)["state"] == "BLOCKED"
    queue.fields = None
    queue.prompt_tag["status"] = "consumed"
    result = recover(coordinator, queue)
    assert result["state"] == "BLOCKED"
    assert queue.put_calls == 1 and queue.prompt_put_calls == 1
    assert "finish" not in coordinator.calls and "gc" not in coordinator.calls


def test_durable_ready_proof_can_finish_after_legal_consumption_without_new_get():
    coordinator = FakeCoordinator()
    coordinator.confirm_response_fail_on = 2
    queue = FakeQueue(coordinator)
    assert run_publish(coordinator, queue)["state"] == "BLOCKED"
    queue.fields = None
    queue.prompt_tag["status"] = "consumed"
    result = recover(coordinator, queue, validator=None)
    assert result["state"] == "FINISHED"
    assert result["readback_proof_reused"] and not result["readback_performed"]
    assert queue.get_calls == 1 and queue.prompt_put_calls == 1


def test_unfinished_ready_proof_with_missing_or_regressed_uid_does_not_gc():
    for status in (None, "running", "unknown"):
        coordinator = FakeCoordinator()
        coordinator.finish_failures = 1
        queue = FakeQueue(coordinator)
        assert run_publish(coordinator, queue)["state"] == "BLOCKED"
        queue.prompt_tag = None if status is None else {"status": status}
        result = recover(coordinator, queue, validator=None)
        assert result["state"] == "BLOCKED"
        assert queue.put_calls == 1 and queue.prompt_put_calls == 1
        assert "gc" not in coordinator.calls


def test_duplicate_finished_uses_durable_ready_receipt_without_queue_writes():
    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator)
    assert run_publish(coordinator, queue)["state"] == "FINISHED"
    queue.fields = None
    queue.prompt_tag = None  # The original consumer may have removed its keys.
    old_calls = list(coordinator.calls)
    result = recover(coordinator, queue, validator=None)
    assert result["state"] == "FINISHED" and result["duplicate_finished"]
    assert not result["readback_performed"]
    assert queue.put_calls == 1 and queue.prompt_put_calls == 1
    assert coordinator.calls == old_calls


def test_legacy_finished_without_ready_proof_is_not_blindly_republished():
    coordinator = FakeCoordinator()
    queue = FakeQueue(coordinator)
    assert run_publish(coordinator, queue)["state"] == "FINISHED"
    del coordinator.publication["receipt"]["backend_reference"]["prompt_ready"]
    result = recover(coordinator, queue, validator=None)
    assert result["state"] == "BLOCKED"
    assert queue.put_calls == 1 and queue.prompt_put_calls == 1


@pytest.mark.skipif(os.environ.get("RUN_TQ_PUBLICATION_RUNTIME_SMOKE") != "1", reason="Explicit isolated CodeLab TQ runtime smoke only")
def test_real_tq_public_batch_put_get_readback_on_cpu():
    import ray
    import transfer_queue as tq
    from omegaconf import OmegaConf

    if ray.is_initialized():
        pytest.skip("Real TQ smoke needs an isolated CPU process; do not reuse a running benchmark cluster")
    partition = "publication-smoke-" + uuid4().hex
    coordinator = FakeCoordinator()  # Runtime TQ proof, not durable coordinator proof.
    try:
        ray.init(address="local", num_cpus=4, num_gpus=0, include_dashboard=False, object_store_memory=80 * 1024 * 1024)
        tq.init(OmegaConf.create({"backend": {"storage_backend": "SimpleStorage", "SimpleStorage": {"num_data_storage_units": 1, "total_storage_size": 32}}}))
        async def smoke():
            await tq.async_kv_put(key=UID, partition_id=partition, tag={"status": "running"})
            fields = original_fields()
            assert fields.get("prompts").is_nested and fields.get("responses").is_nested
            assert len({row.numel() for row in fields.get("prompts").unbind(0)}) > 1
            assert len({row.numel() for row in fields.get("responses").unbind(0)}) > 1
            result = await publish_trajectory_group(
                coordinator=coordinator, handoff_id="fixture-handoff", members=MEMBERS,
                fields=fields, tags=original_tags(), partition_id=partition,
                validate_fields=original_validator, tq_api=tq,
            )
            assert result["state"] == "FINISHED" and result["prompt_ready_verified"]
            assert result["proof_layer"] == "actual_transferqueue_runtime"
            assert result["receipt"]["backend_reference"]["verified_members"] == 8
            assert result["receipt"]["backend_reference"]["tq_version"] == "0.1.10"
            await tq.async_kv_clear(keys=[UID, *[member.key for member in MEMBERS]], partition_id=partition)

        asyncio.run(smoke())
    finally:
        tq.close()
        ray.shutdown()
