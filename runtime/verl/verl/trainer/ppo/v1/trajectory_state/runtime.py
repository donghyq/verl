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

"""Opt-in V1 prompt-group execution, using existing handoff contracts.

Callbacks construct/run the real AgentLoop. This helper does not render tokens,
run rewards, postprocess outputs, publish TransferQueue data, choose a replica,
or acknowledge a Trainer update. Only a committed Manifest guarantees retained
member outcomes after restart. Ordinary complete groups use the stock TQ path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

from .handoff import HandoffManifest, decode_handoff_member_record, encode_handoff_member_record
from .identity import _decode_policy_identity
from .policy_publication import HandoffPublicationBarrier, PublishedPolicyRecord
from .preemptible_tool_agent_loop import TrajectoryCheckpointYield, TrajectoryPendingYield
from .snapshot import TrajectoryOwnerLease, TrajectorySnapshot

logger = logging.getLogger(__name__)

SourceCallback = Callable[[int, dict, TrajectoryOwnerLease, Callable, Callable], Awaitable[Any]]
ResumeCallback = Callable[[int, TrajectorySnapshot, dict, TrajectoryOwnerLease, Callable], Awaitable[Any]]
PendingCallback = Callable[[int, dict, TrajectoryOwnerLease, Callable], Awaitable[Any]]


@dataclass(frozen=True)
class PromptGroupResult:
    status: str  # completed | handed_off
    outputs: list[Any]
    handoff_id: str | None
    member_kwargs: list[dict]
    output_record_refs: dict[int, str]
    metrics: dict
    input_record_refs: dict[int, str] = field(default_factory=dict)
    source_generation: int | None = None


class PromptGroupExecutionError(RuntimeError):
    """Explicit failure/fallback; diagnostic records are not a committed group."""

    def __init__(self, reason: str, *, uid: str, handoff_id: str | None, status: dict,
                 unsettled: list[str] | None = None):
        super().__init__(f"{reason}: prompt group {uid}")
        self.reason, self.uid, self.handoff_id = reason, uid, handoff_id
        self.status, self.unsettled = status, unsettled or []


@dataclass
class _Member:
    index: int
    trajectory_id: str
    kwargs: dict
    input_ref: str | None = None
    status: str = "pending"
    lease: TrajectoryOwnerLease | None = None
    tracker_registered: bool = False
    snapshot_ref: str | None = None
    output_ref: str | None = None
    output: Any = None
    task: asyncio.Task | None = None
    heartbeat_task: asyncio.Task | None = None
    heartbeat_error: BaseException | None = None


@dataclass
class _SourceGroup:
    uid: str
    members: list[_Member]
    producer_generation: int
    signal: asyncio.Event = field(default_factory=asyncio.Event)
    preempt_requested_at: float | None = None
    handoff_id: str | None = None
    policy_barrier: HandoffPublicationBarrier | None = None
    policy_barrier_ready: asyncio.Event = field(default_factory=asyncio.Event)
    policy_barrier_error: BaseException | None = None


class PromptGroupHandoffRuntime:
    def __init__(self, client, *, owner_id: str, incarnation_id: str, policy_identity: dict,
                 environment_version: str, tool_version: str, lease_seconds: float = 600,
                 snapshot_ttl_seconds: float = 900, preemption_deadline_s: float = 60,
                 published_policy_client=None):
        for name, value in {"owner_id": owner_id, "incarnation_id": incarnation_id,
                            "environment_version": environment_version, "tool_version": tool_version}.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        for value in (lease_seconds, snapshot_ttl_seconds, preemption_deadline_s):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("runtime durations must be finite and positive")
        _decode_policy_identity(policy_identity)
        self.client, self.owner_id, self.incarnation_id = client, owner_id, incarnation_id
        self.policy_identity = deepcopy(policy_identity)
        self.environment_version, self.tool_version = environment_version, tool_version
        self.lease_seconds, self.snapshot_ttl_seconds = lease_seconds, snapshot_ttl_seconds
        self.preemption_deadline_s = preemption_deadline_s
        self.published_policy_client = published_policy_client
        self._groups: dict[str, _SourceGroup] = {}
        self._starting_sources: set[str] = set()
        self._recent_status: OrderedDict[str, dict] = OrderedDict()
        self._active_targets: set[str] = set()
        self._publication_finished: set[str] = set()
        self._normal_cleanup_pending: dict[str, dict] = {}

    @staticmethod
    def _status(group: _SourceGroup) -> dict:
        categories = ("pending", "starting", "running", "completed", "checkpointed", "failed")
        return {
            "uid": group.uid, "preemption_requested": group.signal.is_set(), "handoff_id": group.handoff_id,
            "source_generation": group.producer_generation,
            "publication_finished": False,
            **{category + "_members": [member.index for member in group.members if member.status == category]
               for category in categories},
            "completed_record_refs": {member.index: member.output_ref for member in group.members
                                      if member.output_ref is not None},
            "input_record_refs": {member.index: member.input_ref for member in group.members
                                  if member.input_ref is not None},
            "snapshot_refs": {member.index: member.snapshot_ref for member in group.members
                              if member.snapshot_ref is not None},
            "policy_barrier_unsettled": group.policy_barrier is not None,
        }

    def group_status(self, uid: str) -> dict | None:
        """Read-only driver observation; no replicated scheduling state is added."""
        if uid in self._groups:
            return deepcopy(self._status(self._groups[uid]))
        status = self._recent_status.get(uid)
        if status is None:
            return None
        status = deepcopy(status)
        status["publication_finished"] = uid in self._publication_finished
        return status

    async def _resolve_published_policy(self, version) -> dict:
        if self.published_policy_client is None:
            raise ValueError("continuous policies require a publication client")
        record = PublishedPolicyRecord.decode(await self.published_policy_client.get_published_policy(version))
        if (record.identity.tokenizer_template_fingerprint != self.policy_identity["tokenizer_template_fingerprint"]
                or record.identity.kv_cache_schema_fingerprint != self.policy_identity["kv_cache_schema_fingerprint"]):
            raise ValueError("published tokenizer/KV configuration differs from the runtime")
        return record.encode()

    def _attach_policy_checks(self, check_owner, *, target_identity=None, token_only=False):
        if self.published_policy_client is not None:
            check_owner.resolve_published_policy = self._resolve_published_policy
        if token_only:
            check_owner.token_only_resume = True
            check_owner.target_policy_identity = deepcopy(target_identity)

    def _start_lease_heartbeat(self, member: _Member) -> None:
        """Start one bounded renewal loop for a member that owns a lease."""
        if member.lease is None or member.heartbeat_task is not None:
            raise ValueError("member lease heartbeat cannot start")
        member.heartbeat_error = None

        async def renew_until_stopped():
            # The first renewal is one third into the lease, leaving time for
            # one in-flight RPC before the authoritative deadline expires.
            interval = max(self.lease_seconds / 3, 0.001)
            while True:
                await asyncio.sleep(interval)
                try:
                    member.lease = await asyncio.wait_for(
                        self.client.renew_trajectory_owner(member.lease, self.lease_seconds),
                        timeout=min(interval, 30),
                    )
                except asyncio.CancelledError:
                    raise
                except BaseException as error:
                    # A lost background failure must fail the owning callback;
                    # it must not silently let execution continue without rights.
                    member.heartbeat_error = error
                    task = member.task
                    if task is not None and not task.done():
                        task.cancel()
                    return

        member.heartbeat_task = asyncio.create_task(renew_until_stopped())

    async def _stop_lease_heartbeat(self, member: _Member) -> None:
        """Stop renewal and surface any error that ended it."""
        task, member.heartbeat_task = member.heartbeat_task, None
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        error, member.heartbeat_error = member.heartbeat_error, None
        if error is not None:
            raise error

    async def _cancel_lease_heartbeat(self, member: _Member) -> None:
        """Stop renewal during fencing without replacing the fencing error."""
        task, member.heartbeat_task = member.heartbeat_task, None
        member.heartbeat_error = None
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def wait_for_preemption(self, uid: str) -> bool:
        """Wake a live source stream even when its engine has no next SSE row."""
        group = self._groups.get(uid)
        if group is None:
            return False
        await group.signal.wait()
        return True

    async def request_preemption(self, uid: str) -> dict | bool:
        group = self._groups.get(uid)
        if group is None:
            return False  # Manager may broadcast to workers that do not own uid.
        if not group.signal.is_set():
            group.preempt_requested_at = time.monotonic()
            group.signal.set()  # Stop new admissions before any coordinator await.
            if self.published_policy_client is not None:
                try:
                    leases = [member.lease for member in group.members if member.lease is not None
                              and member.status in {"starting", "running", "checkpointed"}]
                    group.policy_barrier = await self.published_policy_client.begin_handoff_policy_barrier(
                        uid, leases, self.preemption_deadline_s)
                except BaseException as error:
                    group.policy_barrier_error = error
                    raise
                finally:
                    group.policy_barrier_ready.set()
        # A member still registering checks the signal itself before execution.
        for member in group.members:
            if member.status in {"starting", "running"} and member.tracker_registered:
                try:
                    await self.client.request_owned_trajectory_preemption(member.lease)
                except ValueError:
                    # A member can legitimately finish/release while the
                    # signal RPC is in flight. A still-running stale owner is
                    # an actual error and cannot be accepted as a control write.
                    if member.lease is None or member.status in {"completed", "pending", "checkpointed"}:
                        continue
                    raise
        return self._status(group)

    @staticmethod
    def _record(uid, n, member, kind, *, output=None) -> bytes:
        value = {"trajectory_runtime_record": 1, "kind": kind, "uid": uid,
                 "member_index": member.index, "group_size": n,
                 "trajectory_id": member.trajectory_id, "member_kwargs": member.kwargs}
        if kind == "output":
            value["output"] = output
        return encode_handoff_member_record(value)

    async def _read_record(self, reference, *, kind, uid, n, index, trajectory_id) -> dict:
        record = decode_handoff_member_record(await self.client.get_handoff_member_record(reference))
        expected = {"trajectory_runtime_record": 1, "kind": kind, "uid": uid,
                    "member_index": index, "group_size": n, "trajectory_id": trajectory_id}
        if (not isinstance(record, dict) or any(record.get(key) != value for key, value in expected.items())
                or not isinstance(record.get("member_kwargs"), dict)
                or (kind == "output" and "output" not in record)):
            raise ValueError("durable member record identity or kind differs")
        return record

    async def _write_member_lifecycle_receipt(self, manifest, *, uid, n, refs, target_identity):
        """Read back original member records; do not turn completion into KV recovery.

        This opt-in diagnostic uses the existing receipt directory. The final
        driver still checks all-rank physical restore records independently.
        """
        directory = os.environ.get("VERL_TRAJECTORY_RECEIPT_DIR")
        if not directory:
            return
        categories = (("completed", manifest.completed_members),
                      ("checkpointed", manifest.checkpointed_trajectories),
                      ("pending", manifest.pending_samples))
        rows = []
        for category, members in categories:
            for member in members:
                index, trajectory_id = member["member_index"], member["trajectory_id"]
                reference = refs[index]
                if category == "completed" and reference != member["record_ref"]:
                    raise ValueError("completed member must retain its original durable result")
                input_ref = (member.get("input_ref") if category == "checkpointed"
                             else member["record_ref"] if category == "pending" else None)
                if input_ref is not None:
                    await self._read_record(input_ref, kind="input", uid=uid, n=n,
                                            index=index, trajectory_id=trajectory_id)
                raw = await self.client.get_handoff_member_record(reference)
                # Reuse the same trusted codec and identity contract as _read_record.
                record = decode_handoff_member_record(raw)
                expected = {"trajectory_runtime_record": 1, "kind": "output", "uid": uid,
                            "member_index": index, "group_size": n, "trajectory_id": trajectory_id}
                if (not isinstance(record, dict) or any(record.get(k) != v for k, v in expected.items())
                        or "output" not in record or not isinstance(record.get("member_kwargs"), dict)):
                    raise ValueError("lifecycle output record identity or kind differs")
                original, prepared = record["member_kwargs"], record["output"]
                if (original.get("uid") != uid or original.get("session_id") != index
                        or original.get("__trajectory_group_size__") != n
                        or getattr(prepared, "keys", None) != [f"{uid}_{index}_0"]):
                    raise ValueError("lifecycle member differs from its original V1 member")
                output = getattr(prepared, "output", None)
                extra = getattr(output, "extra_fields", {})
                identity = {"trajectory_id": trajectory_id, "rollout_group_id": uid,
                            "rollout_sample_index": index, "rollout_group_size": n}
                generation = extra.get("trajectory_execution_generation")
                if (any(extra.get(k) != v for k, v in identity.items())
                        or type(generation) is not int or generation < 0):
                    raise ValueError("completed output lacks its actual member execution identity")
                mask = getattr(output, "response_mask", None)
                tokens = getattr(output, "response_ids", None)
                segments = extra.get("trajectory_segments")
                if (not isinstance(mask, list) or not isinstance(tokens, list) or len(mask) != len(tokens)
                        or any(type(x) is not int or x not in (0, 1) for x in mask)
                        or not isinstance(segments, list) or not segments):
                    raise ValueError("completed output lacks generated-token policy segments")
                cursor, versions = 0, set()
                for segment in segments:
                    start, end, version = segment.get("start"), segment.get("end"), segment.get("policy_version")
                    if (type(start) is not int or type(end) is not int or start != cursor
                            or end <= start or end > sum(mask) or not isinstance(version, str) or not version):
                        raise ValueError("completed output has incomplete actual policy segments")
                    published = segment.get("policy_identity")
                    if published is not None and published not in (manifest.policy_identity, target_identity):
                        raise ValueError("completed output has a different published policy identity")
                    cursor = end
                    versions.add(version)
                if cursor != sum(mask):
                    raise ValueError("completed output policy segments do not cover actual generated tokens")
                recovery = extra.get("trajectory_token_kv_recovery") or {}
                checkpoint = None
                if category == "checkpointed":
                    # The existing immutable-object read accepts the manifest's
                    # snapshot reference too; no owner or execution is acquired.
                    snapshot_raw = await self.client.get_handoff_member_record(member["snapshot_ref"])
                    saved = json.loads(snapshot_raw)
                    expected_snapshot = {"trajectory_id": trajectory_id, "group_id": uid,
                        "member_index": index, "group_size": n,
                        "execution_generation": member["expected_generation"],
                        "policy_identity": manifest.policy_identity,
                        "environment_version": manifest.environment_version, "tool_version": manifest.tool_version}
                    if not isinstance(saved, dict) or any(saved.get(k) != v for k, v in expected_snapshot.items()):
                        raise ValueError("committed checkpoint differs from its original member")
                    kv = saved.get("kv_ref")
                    if saved.get("schema_version") == 1 and kv is None:
                        # Only the existing complete-tool-boundary contract can
                        # establish that no current generation/KV needs restore.
                        tool_snapshot = TrajectorySnapshot.decode(snapshot_raw)
                        checkpoint = {"kind": "TOOL_BOUNDARY", "schema_version": 1,
                            "next_stage": tool_snapshot.next_stage, "kv_ref_absent": True,
                            "completed_tool_batches": len(tool_snapshot.completed_tool_batches),
                            "retained_generated_tokens": sum(tool_snapshot.agent_state["response_mask"])}
                    elif (saved.get("schema_version") == 2 and saved.get("next_stage") == "generating_inflight"
                          and isinstance(kv, dict) and kv.get("kind") == "token_kv_committed_v1"
                          and kv.get("snapshot_committed") is True):
                        checkpoint = {"kind": "TOKEN_KV", "schema_version": 2,
                            "next_stage": "generating_inflight", "snapshot_id": kv.get("snapshot_id")}
                    else:
                        raise ValueError("checkpoint has no supported committed restore kind")
                    checkpoint["snapshot_sha256"] = hashlib.sha256(snapshot_raw).hexdigest()
                rows.append({
                    "member_index": index, "trajectory_id": trajectory_id, "source_state": category,
                    "output_record_ref": reference, "output_record_sha256": hashlib.sha256(raw).hexdigest(),
                    "input_record_ref": input_ref, "snapshot_ref": member.get("snapshot_ref"),
                    "source_execution_generation": member.get("expected_generation"),
                    "output_execution_generation": generation, "actual_policy_versions": sorted(versions),
                    "recovery_mode": recovery.get("mode"), "snapshot_id": recovery.get("snapshot_id"),
                    "checkpoint": checkpoint,
                })
        if sorted(row["member_index"] for row in rows) != list(range(n)):
            raise ValueError("lifecycle receipt must cover every original member exactly once")
        receipt = {"schema": 1, "status": "VERIFIED_DURABLE_MEMBER_RECORDS",
                   "uid": uid, "handoff_id": manifest.handoff_id, "group_size": n,
                   "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
                   "manifest_source_generation": manifest.source_generation,
                   "source_policy_identity": manifest.policy_identity, "target_policy_identity": target_identity,
                   "members": sorted(rows, key=lambda row: row["member_index"])}
        path = Path(directory) / ("member-lifecycle-" + hashlib.sha256(manifest.handoff_id.encode()).hexdigest()[:16] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    async def _source_member(self, group, member, run_member):
        if group.signal.is_set():
            return
        member.status = "starting"
        member.lease = await self.client.open_trajectory(member.trajectory_id, self.owner_id, self.lease_seconds)
        await self.client.register_owned_running_trajectory(member.lease)
        member.tracker_registered = True
        if group.signal.is_set():
            # No callback/tool/model has run; the exact original input stays pending.
            await self.client.release_trajectory_owner(member.lease)
            member.lease, member.status = None, "pending"
            return
        member.status = "running"

        async def check_owner():
            await self.client.check_trajectory_owner(member.lease)

        self._attach_policy_checks(check_owner)

        async def get_snapshot_policy_identity():
            if not group.signal.is_set():
                return None
            await group.policy_barrier_ready.wait()
            if group.policy_barrier_error is not None:
                raise group.policy_barrier_error
            return await self.published_policy_client.check_handoff_policy_barrier(group.policy_barrier)

        async def commit_snapshot(snapshot):
            await check_owner()
            if not group.signal.is_set():
                return None
            source_identity = self.policy_identity
            if self.published_policy_client is not None:
                source_identity = await get_snapshot_policy_identity()
            state = await self.client.get_trajectory_preemption(member.trajectory_id)
            if state == "running":
                await self.client.request_owned_trajectory_preemption(member.lease)
                state = "preempt_requested"
            if state != "preempt_requested":
                raise ValueError("snapshot callback does not have a current preemption request")
            if (snapshot.trajectory_id != member.trajectory_id or snapshot.group_id != group.uid
                    or snapshot.member_index != member.index or snapshot.group_size != len(group.members)
                    or snapshot.execution_generation != member.lease.generation
                    or snapshot.policy_identity != source_identity
                    or snapshot.environment_version != self.environment_version
                    or snapshot.tool_version != self.tool_version):
                raise ValueError("source snapshot identity differs from the prompt member")
            await self.client.mark_owned_trajectory_safe_point(member.lease)
            reference = await self.client.commit_trajectory_snapshot(member.lease, snapshot, self.snapshot_ttl_seconds)
            member.snapshot_ref = reference
            return reference

        async def commit_unstarted_generation(receipt):
            await check_owner()
            if not group.signal.is_set():
                raise ValueError("pending cancellation requires current source preemption")
            if self.published_policy_client is not None:
                await get_snapshot_policy_identity()
            # Reuse the existing immutable input object. The existing handoff
            # manifest will reference this replacement input record; original
            # kwargs/tokens are preserved, not synthesized from a zero SSE count.
            record = await self._read_record(member.input_ref, kind="input", uid=group.uid,
                n=len(group.members), index=member.index, trajectory_id=member.trajectory_id)
            record["cancelled_native_generation"] = deepcopy(receipt)
            reference = await self.client.put_handoff_member_record(encode_handoff_member_record(record))
            await check_owner()
            member.input_ref = reference
            return reference

        check_owner.commit_unstarted_generation = commit_unstarted_generation

        if self.published_policy_client is not None:
            commit_snapshot.resolve_published_policy = self._resolve_published_policy
            commit_snapshot.get_snapshot_policy_identity = get_snapshot_policy_identity

        self._start_lease_heartbeat(member)
        try:
            output = await run_member(member.index, deepcopy(member.kwargs), member.lease, commit_snapshot, check_owner)
        except TrajectoryCheckpointYield as yielded:
            await self._stop_lease_heartbeat(member)
            if member.snapshot_ref is None or yielded.reference != member.snapshot_ref:
                raise ValueError("AgentLoop yielded without this runtime's durable snapshot commit") from yielded
            member.status = "checkpointed"
            return
        except TrajectoryPendingYield as yielded:
            await self._stop_lease_heartbeat(member)
            if not group.signal.is_set() or yielded.trajectory_id != member.trajectory_id or member.snapshot_ref is not None:
                raise ValueError("pending yield must belong to this preempted, unsnapshotted member") from yielded
            await check_owner()
            await self.client.release_trajectory_owner(member.lease)
            member.lease, member.status = None, "pending"
            return
        except BaseException:
            await self._stop_lease_heartbeat(member)
            raise
        await self._stop_lease_heartbeat(member)
        await check_owner()
        member.output_ref = await self.client.put_handoff_member_record(
            self._record(group.uid, len(group.members), member, "output", output=output))
        await check_owner()
        await self.client.finish_trajectory(member.lease)
        member.lease, member.output, member.status = None, output, "completed"

    async def _wait_source(self, group, run_member, initial):
        def launch(indices):
            for index in indices:
                member = group.members[index]
                member.task = asyncio.create_task(self._source_member(group, member, run_member))

        launch(initial)
        signal_wait = asyncio.create_task(group.signal.wait())
        try:
            while True:
                tasks = [member.task for member in group.members if member.task is not None]
                for task in tasks:
                    if task.done():
                        task.result()  # Fail directly, rather than losing errors in a wait list.
                active = [task for task in tasks if not task.done()]
                if not active:
                    if group.signal.is_set():
                        return
                    pending = [m.index for m in group.members if m.status == "pending" and m.task is None]
                    if not pending:
                        return
                    # start_members only controls a deterministic driver fixture.
                    # Normal execution admits all members; no routing policy exists.
                    launch(pending)
                    continue
                timeout = None
                if group.signal.is_set():
                    timeout = max(0, group.preempt_requested_at + self.preemption_deadline_s - time.monotonic())
                waits = set(active)
                if not signal_wait.done():
                    waits.add(signal_wait)
                done, _ = await asyncio.wait(waits, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    raise PromptGroupExecutionError("SAFE_POINT_TIMEOUT", uid=group.uid,
                                                    handoff_id=group.handoff_id, status=self._status(group))
        finally:
            signal_wait.cancel()
            await asyncio.gather(signal_wait, return_exceptions=True)

    async def _fence_and_cancel(self, members) -> list[str]:
        """Fence only this invocation's rights, then cancel its callback tasks."""
        unsettled = []
        for member in members:
            await self._cancel_lease_heartbeat(member)
        for member in members:
            if member.lease is None or member.status == "checkpointed":
                continue  # A committed snapshot must not be aborted as if no work survived.
            try:
                result = await self.client.abort_owned_trajectory(member.lease)
                if result.get("released") is not True:
                    unsettled.append(member.trajectory_id)
                    continue
                member.lease = None
                if member.status != "completed":
                    member.status = "failed"
            except ValueError:
                # The coordinator already invalidated these rights (expiry or
                # restart). Never mutate another generation's preemption state.
                member.lease = None
                continue
            except Exception:
                unsettled.append(member.trajectory_id)
        all_tasks = [m.task for m in members if m.task is not None]
        tasks = [task for task in all_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=1)
            for member in members:
                if member.task in pending:
                    unsettled.append(member.trajectory_id)
        # Simultaneous sibling failures may already be done before fencing.
        # Retrieve every settled exception without waiting beyond the deadline.
        for task in all_tasks:
            if task.done():
                try:
                    task.result()
                except (Exception, asyncio.CancelledError):
                    pass
        return sorted(set(unsettled))

    def _result(self, group, *, status, outputs, elapsed):
        return PromptGroupResult(
            status=status, outputs=outputs, handoff_id=group.handoff_id,
            member_kwargs=[deepcopy(member.kwargs) for member in group.members],
            output_record_refs={m.index: m.output_ref for m in group.members if m.output_ref is not None},
            input_record_refs={m.index: m.input_ref for m in group.members if m.input_ref is not None},
            source_generation=group.producer_generation,
            metrics={"group_completion_s": elapsed if status == "completed" else None,
                     "source_execution_release_s": time.monotonic() - group.preempt_requested_at
                     if group.preempt_requested_at is not None and status == "handed_off" else None,
                     "worker_reclaim_s": None,  # A driver must observe actual resource reclamation.
                     "safe_point_wait_s": None, "snapshot_commit_s": None, "takeover_s": None,
                     "resume_to_first_token_s": None, "preserved_generated_tokens": None,
                     "lost_generated_tokens": None, "recomputed_prefill_tokens": None,
                     "tool_replay_count": None},
        )

    async def run_source(self, uid: str, member_kwargs: list[dict], run_member: SourceCallback, *,
                         start_members: list[int] | tuple[int, ...] | None = None) -> PromptGroupResult:
        """Run one prompt group; commit/release before returning a handoff ID.

        ``run_member(i, original_kwargs, lease, commit_snapshot, check_owner)``
        calls the registered tool adapter with the supplied callbacks. Inputs
        include original sampling_params/trajectory values and are not rebuilt.
        ``start_members`` is a driver fixture only, never a production scheduler.
        """
        if self._normal_cleanup_pending:
            await self.retry_normal_cleanup()
        if len(self._normal_cleanup_pending) >= 128:
            raise RuntimeError("normal trajectory cleanup backlog reached its admission limit")
        if (not isinstance(uid, str) or not uid or uid in self._groups
                or uid in self._starting_sources
                or self._recent_status.get(uid, {}).get("handoff_id") is not None
                or not isinstance(member_kwargs, list)
                or not member_kwargs or any(not isinstance(value, dict) for value in member_kwargs)
                or not callable(run_member)):
            raise ValueError("invalid or already-active prompt group")
        n = len(member_kwargs)
        initial = list(range(n)) if start_members is None else list(start_members)
        if (not initial or any(type(index) is not int or not 0 <= index < n for index in initial)
                or len(set(initial)) != len(initial)):
            raise ValueError("start_members must contain unique valid member indices")
        # Reserve before the first RPC so concurrent calls cannot overwrite the
        # worker's active group even while actor registration is in flight.
        self._starting_sources.add(uid)
        try:
            source_generation = await self.client.register_producer(self.owner_id, self.incarnation_id)
        except BaseException:
            self._starting_sources.discard(uid)
            raise
        group = _SourceGroup(uid, [_Member(i, f"{uid}/member-{i}", deepcopy(value))
                                   for i, value in enumerate(member_kwargs)], source_generation)
        self._groups[uid] = group
        self._starting_sources.discard(uid)
        started = time.monotonic()
        try:
            for member in group.members:
                member.input_ref = await self.client.put_handoff_member_record(self._record(uid, n, member, "input"))
            await self._wait_source(group, run_member, initial)
            if all(member.status == "completed" for member in group.members):
                return self._result(group, status="completed", outputs=[m.output for m in group.members],
                                    elapsed=time.monotonic() - started)
            if not group.signal.is_set() or not any(m.status == "checkpointed" for m in group.members):
                raise PromptGroupExecutionError("NO_SAFE_POINT_SNAPSHOT", uid=uid,
                                                handoff_id=None, status=self._status(group))
            if any(m.status not in {"completed", "checkpointed", "pending"} for m in group.members):
                raise ValueError("source group did not quiesce at complete/committed/pending states")
            completed, checkpointed, pending = [], [], []
            for member in group.members:
                base = {"trajectory_id": member.trajectory_id, "group_id": uid,
                        "member_index": member.index, "group_size": n}
                if member.status == "completed":
                    completed.append({**base, "record_ref": member.output_ref})
                elif member.status == "checkpointed":
                    checkpointed.append({**base, "snapshot_ref": member.snapshot_ref,
                                         "expected_generation": member.lease.generation, "input_ref": member.input_ref})
                else:
                    pending.append({**base, "record_ref": member.input_ref})
            # Group identity remains in Manifest membership. A resource-level
            # handoff ID follows the unchanged registry's path-safe contract.
            uid_digest = hashlib.sha256(uid.encode("utf-8")).hexdigest()[:16]
            group.handoff_id = f"handoff-{uid_digest}-{uuid4().hex}"
            manifest = HandoffManifest(
                group.handoff_id, self.owner_id, source_generation,
                vars(group.policy_barrier.identity).copy() if group.policy_barrier is not None
                else deepcopy(self.policy_identity),
                self.environment_version, self.tool_version, (), tuple(checkpointed), tuple(completed), tuple(pending),
            )
            await self.client.prepare_handoff_manifest(manifest)
            await self.client.commit_handoff_manifest(group.handoff_id)
            await self.client.release_handoff_source(group.handoff_id)
            for member in group.members:
                if member.status == "checkpointed":
                    member.lease = None
            if group.policy_barrier is not None:
                await self.published_policy_client.release_handoff_policy_barrier(
                    group.policy_barrier, group.handoff_id
                )
                group.policy_barrier = None
            return self._result(group, status="handed_off", outputs=[], elapsed=time.monotonic() - started)
        except asyncio.CancelledError:
            await self._fence_and_cancel(group.members)
            raise
        except Exception as error:
            unsettled = await self._fence_and_cancel(group.members)
            if isinstance(error, PromptGroupExecutionError):
                error.unsettled = unsettled
                raise
            raise PromptGroupExecutionError("SOURCE_EXECUTION_FAILED", uid=uid, handoff_id=group.handoff_id,
                                            status=self._status(group), unsettled=unsettled) from error
        finally:
            if group.policy_barrier is not None:
                try:
                    # Failure may have retained a committed Snapshot. Revoke
                    # only this source's actual rights, without deleting its
                    # legal history or marking the Snapshot aborted. Each RPC
                    # checks the original lease atomically; a new owner is safe.
                    for member in group.members:
                        if member.lease is not None:
                            try:
                                await self.client.release_trajectory_owner(member.lease)
                                member.lease = None
                            except ValueError:
                                pass  # Old rights are already fenced; never release by trajectory ID.
                    await self.published_policy_client.release_handoff_policy_barrier(group.policy_barrier)
                    group.policy_barrier = None
                except Exception:
                    pass  # Remains visible and blocks new weight publication; no timer-based reopening.
            self._recent_status[uid] = self._status(group)
            while len(self._recent_status) > 128:
                expired_uid, _ = self._recent_status.popitem(last=False)
                self._publication_finished.discard(expired_uid)
            self._groups.pop(uid, None)

    @staticmethod
    def _normal_cleanup_request(result: PromptGroupResult) -> dict:
        if (result.status != "completed" or result.handoff_id is not None
                or type(result.source_generation) is not int or result.source_generation < 0):
            raise ValueError("normal cleanup requires a completed source generation")
        expected = set(range(8))
        refs = {}
        for name in ("input_record_refs", "output_record_refs"):
            values = getattr(result, name)
            if (not isinstance(values, dict) or set(values) != expected
                    or any(not isinstance(value, str) or not value for value in values.values())):
                raise ValueError("normal cleanup requires all eight input and output records")
            refs[name] = dict(values)
        if len(set(refs["input_record_refs"].values()) | set(refs["output_record_refs"].values())) != 16:
            raise ValueError("normal cleanup requires sixteen distinct member records")
        uids = {member.get("uid") for member in result.member_kwargs
                if isinstance(member, dict) and isinstance(member.get("uid"), str) and member.get("uid")}
        if (len(result.member_kwargs) != 8 or len(uids) != 1
                or any(not isinstance(member, dict) or member.get("uid") not in uids
                       for member in result.member_kwargs)):
            raise ValueError("normal cleanup requires one consistent prompt UID")
        return {
            "uid": next(iter(uids)),
            "owner_id": None,
            "source_generation": result.source_generation,
            "input_record_refs": refs["input_record_refs"],
            "output_record_refs": refs["output_record_refs"],
        }

    async def cleanup_completed_group(self, result: PromptGroupResult) -> int:
        """Queue and collect only this normal group's verified member records."""
        request = self._normal_cleanup_request(result)
        request["owner_id"] = self.owner_id
        self._normal_cleanup_pending[request["uid"]] = request
        queued = await self.client.queue_completed_group_cleanup(
            uid=request["uid"], owner_id=self.owner_id,
            source_generation=request["source_generation"],
            input_record_refs=request["input_record_refs"],
            output_record_refs=request["output_record_refs"],
        )
        # The coordinator now durably owns the retry list. A delete failure
        # must not retain another unbounded copy in this worker's memory.
        self._normal_cleanup_pending.pop(request["uid"], None)
        await self.client.collect_garbage()
        return queued

    async def retry_normal_cleanup(self) -> int:
        queued_count = 0
        for uid, request in list(self._normal_cleanup_pending.items()):
            try:
                queued_count += await self.client.queue_completed_group_cleanup(**request)
            except Exception:
                logger.warning("Normal trajectory cleanup queue retry failed: uid=%s", uid, exc_info=True)
                continue
            self._normal_cleanup_pending.pop(uid, None)
        return queued_count

    async def confirm_normal_publication(self, result: PromptGroupResult) -> bool:
        """Confirm a successful original-UID finished PUT and retry cleanup later.

        The caller invokes this only after TransferQueue accepted the prompt's
        finished status. Cleanup is best effort: its failure cannot turn the
        already-published training data into a failure or reissue generation.
        """
        request = self._normal_cleanup_request(result)
        status = self._recent_status.get(request["uid"])
        if status is not None:
            self._publication_finished.add(request["uid"])
            status["publication_finished"] = True
        try:
            await self.cleanup_completed_group(result)
        except Exception:
            logger.warning(
                "Normal trajectory cleanup remains retryable after successful publication: uid=%s",
                request["uid"], exc_info=True,
            )
            return False
        return True

    async def run_target(self, handoff_id: str, resume_member: ResumeCallback,
                         run_pending: PendingCallback, *, token_only: bool = False) -> PromptGroupResult:
        """Resume only unfinished members and return original outputs in index order.

        Existing restored results are read before allocating a new generation.
        Results are persisted before leases are released and before return. TQ
        publication remains the integration owner's separate verified step.
        """
        if (type(token_only) is not bool or handoff_id in self._active_targets
                or not callable(resume_member) or not callable(run_pending)):
            raise ValueError("invalid or already-active target handoff")
        self._active_targets.add(handoff_id)
        members = []
        unsettled = []
        uid = "unknown"
        started = time.monotonic()
        try:
            manifest = await self.client.get_handoff_manifest(handoff_id)
            manifest.validate()
            target_identity = self.policy_identity
            if self.published_policy_client is not None:
                source_record = PublishedPolicyRecord.decode(
                    await self._resolve_published_policy(manifest.policy_identity["policy_version"]))
                if vars(source_record.identity) != manifest.policy_identity:
                    raise ValueError("Manifest source identity differs from actual policy publication")
                target_record = PublishedPolicyRecord.decode(
                    await self.published_policy_client.get_active_published_policy())
                target_identity = vars(target_record.identity).copy()
            elif token_only:
                raise ValueError("new-policy token-only takeover requires a publication client")
            if (manifest.environment_version != self.environment_version or manifest.tool_version != self.tool_version
                    or (not token_only and manifest.policy_identity != target_identity)
                    or manifest.policy_identity["tokenizer_template_fingerprint"]
                    != target_identity["tokenizer_template_fingerprint"]):
                raise ValueError("target environment/tool/policy identity differs")
            tables = manifest.member_tables()
            if len(tables) != 1:
                raise ValueError("runtime only accepts one prompt group")
            uid, table = next(iter(tables.items()))
            n = table["group_size"]
            outputs, refs, kwargs_by_index = {}, {}, {}
            restored = await self.client.get_handoff_restored_members(handoff_id)
            checkpointed = {row["member_index"]: row for row in manifest.checkpointed_trajectories}
            pending = {row["member_index"]: row for row in manifest.pending_samples}
            for row in manifest.completed_members:
                index = row["member_index"]
                record = await self._read_record(row["record_ref"], kind="output", uid=uid, n=n,
                                                 index=index, trajectory_id=row["trajectory_id"])
                outputs[index], refs[index] = record["output"], row["record_ref"]
                kwargs_by_index[index] = record["member_kwargs"]
            for index, row in {**checkpointed, **pending}.items():
                input_ref = row["input_ref"] if index in checkpointed else row["record_ref"]
                original = await self._read_record(input_ref, kind="input", uid=uid, n=n,
                                                   index=index, trajectory_id=row["trajectory_id"])
                kwargs_by_index[index] = original["member_kwargs"]
                if index in restored:
                    reference = restored[index]["record_ref"]
                    record = await self._read_record(reference, kind="output", uid=uid, n=n,
                                                     index=index, trajectory_id=row["trajectory_id"])
                    outputs[index], refs[index] = record["output"], reference
            await self.client.begin_handoff_restore(handoff_id)

            async def run_one(index, row):
                member = _Member(index, row["trajectory_id"], kwargs_by_index[index])
                member.task = asyncio.current_task()
                members.append(member)
                acquire = (self.client.acquire_handoff_trajectory if index in checkpointed
                           else self.client.acquire_handoff_pending)
                member.lease = await acquire(handoff_id, member.trajectory_id, self.owner_id, self.lease_seconds)
                member.status = "running"
                self._start_lease_heartbeat(member)

                async def check_owner():
                    await self.client.check_trajectory_owner(member.lease)

                self._attach_policy_checks(check_owner, target_identity=target_identity, token_only=token_only)

                try:
                    if index in checkpointed:
                        snapshot = await self.client.load_trajectory_snapshot(member.lease)
                        output = await resume_member(index, snapshot, deepcopy(member.kwargs), member.lease, check_owner)
                    else:
                        output = await run_pending(index, deepcopy(member.kwargs), member.lease, check_owner)
                    await self._stop_lease_heartbeat(member)
                    await check_owner()
                    reference = await self.client.record_handoff_member_result(
                        handoff_id, index, member.lease, self._record(uid, n, member, "output", output=output),
                        {"completed_at_unix": time.time(), "resume_to_first_token_s": None, "tool_replay_count": None},
                    )
                    member.output_ref, member.output, member.status = reference, output, "completed"
                    outputs[index], refs[index] = output, reference
                    await self.client.release_trajectory_owner(member.lease)
                    member.lease = None
                except BaseException:
                    await self._stop_lease_heartbeat(member)
                    raise
            tasks = []
            for index, row in sorted({**checkpointed, **pending}.items()):
                if index not in outputs:
                    task = asyncio.create_task(run_one(index, row))
                    tasks.append(task)
            try:
                if tasks:
                    await asyncio.gather(*tasks)
            except BaseException:
                # Fence acquired rights before cancelling callbacks; tasks that
                # have not acquired an owner are cancelled from the complete list.
                unsettled = await self._fence_and_cancel(members)
                for task in tasks:
                    if not task.done():
                        task.cancel()
                raise
            if sorted(outputs) != list(range(n)):
                raise ValueError("restored prompt group has duplicate or missing outputs")
            await self._write_member_lifecycle_receipt(manifest, uid=uid, n=n, refs=refs,
                                                       target_identity=target_identity)
            return PromptGroupResult(
                "completed", [outputs[index] for index in range(n)], handoff_id,
                [kwargs_by_index[index] for index in range(n)], refs,
                {"target_completion_s": time.monotonic() - started, "takeover_s": None,
                 "resume_to_first_token_s": None, "tool_replay_count": None},
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise PromptGroupExecutionError("TARGET_EXECUTION_FAILED", uid=uid, handoff_id=handoff_id,
                                            status={"durable_restored_results_must_be_reused": True},
                                            unsettled=unsettled) from error
        finally:
            self._active_targets.discard(handoff_id)
