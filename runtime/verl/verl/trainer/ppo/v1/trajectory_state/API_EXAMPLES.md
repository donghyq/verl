# V1 trajectory state integration

This module is opt-in. It does not replace V1's ReplayBuffer, TransferQueue,
checkpoint manager, reward functions or optimizer. The coordinator covers
unfinished text ToolAgentLoop trajectories. Each Manifest describes one prompt
group and one final output per member.

## Actor and client

```python
from verl.trainer.ppo.v1.trajectory_state import (
    HandoffManifest,
    RayTrajectoryCoordinatorClient,
    create_ray_trajectory_coordinator_actor,
    decode_handoff_member_record,
    encode_handoff_member_record,
)

Actor = create_ray_trajectory_coordinator_actor(max_restarts=1)
actor = Actor.options(name="trajectory-state-<unique-run>").remote(
    root_dir="/shared/rollout-state",  # verified flock + rename + fsync filesystem
    run_id="unique-run",
)
client = RayTrajectoryCoordinatorClient(actor)
source_generation = await client.register_producer("source-worker", "source-incarnation")
```

The client contains only an ActorHandle and can cross Ray worker boundaries.
Callbacks must capture the client and lease, never a coordinator instance or its
file lock. A coordinator restart invalidates every old execution lease. Producer
generation histories survive; a target receives the next existing generation.

## Source: retain original inputs and yield at the existing tool boundary

Before dispatch, preserve each member's original kwargs needed by V1, including
`uid`, `session_id`, sampling seed and tool kwargs. Do not serialize callbacks or
untrusted executable objects. The cloudpickle envelope is the existing private,
trusted runtime format, not an external deserialization API.

```python
input_refs = {}
for i in range(n):
    input_refs[i] = await client.put_handoff_member_record(
        encode_handoff_member_record(original_member_kwargs[i])
    )

trajectory_id = f"{group_id}#{member_index}"
lease = await client.open_trajectory(trajectory_id, "source-worker", 600.0)
await client.register_owned_running_trajectory(lease)

async def check_owner():
    await client.check_trajectory_owner(lease)

async def commit_at_tool_boundary(snapshot):
    if await client.get_trajectory_preemption(trajectory_id) != "preempt_requested":
        return None
    await client.mark_owned_trajectory_safe_point(lease)
    reference = await client.commit_trajectory_snapshot(lease, snapshot, 3600.0)
    # latest + SNAPSHOT_COMMITTED are one durable transaction. The historical
    # separate marker call remains available and is idempotent for this ref.
    await client.mark_trajectory_snapshot_committed(trajectory_id, reference)
    return reference

# Reuse the existing run_until_checkpoint call and ToolAgentLoop behavior.
# The integration owner binds trajectory identity and the callbacks.
# The driver injects the existing request_preemption signal when authorized.
await client.request_owned_trajectory_preemption(lease)
```

The snapshot object is `trajectory_state.TrajectorySnapshot`; its fields and
canonical JSON bytes are migrated from f91c201d. For completed source members,
save their actual full output using `put_handoff_member_record` before releasing
the source lease. Pending members have an original input record but no running
lease.

## Commit one n=8 mixed group, then release the source

```python
manifest = HandoffManifest(
    handoff_id="handoff-<unique-group>",
    source_rollouter="source-worker",
    source_generation=source_generation,
    policy_identity=policy_identity,  # all four existing canonical identity fields
    environment_version=environment_version,
    tool_version=tool_version,
    completed_artifacts=(),
    completed_members=tuple({
        "trajectory_id": f"{group_id}#{i}", "group_id": group_id,
        "member_index": i, "group_size": 8, "record_ref": completed_output_refs[i],
    } for i in (0, 1)),
    checkpointed_trajectories=tuple({
        "trajectory_id": f"{group_id}#{i}", "group_id": group_id,
        "member_index": i, "group_size": 8, "snapshot_ref": snapshot_refs[i],
        "expected_generation": source_leases[i].generation, "input_ref": input_refs[i],
    } for i in (2, 3, 4, 5)),
    pending_samples=tuple({
        "trajectory_id": f"{group_id}#{i}", "group_id": group_id,
        "member_index": i, "group_size": 8, "record_ref": input_refs[i],
    } for i in (6, 7)),
)
await client.prepare_handoff_manifest(manifest)
await client.commit_handoff_manifest(manifest.handoff_id)
await client.release_handoff_source(manifest.handoff_id)
# Only now may the source CPU executor exit. GPU quota reclaim is separate.
```

`record_handoff_source_release` rejects active source leases. Prefer the atomic
`release_handoff_source`; unlike the historical adapter, do not record release
before releasing the actual write rights.

## Reopen and restore

Use the same `root_dir` and `run_id` after the old Actor process has exited. Ray
automatic restart reconstructs the actor with its original constructor arguments.
Do not create another independent coordinator while the old writer is alive.

```python
manifest = await client.get_handoff_manifest(handoff_id)
await client.begin_handoff_restore(handoff_id)
saved = await client.get_handoff_restored_members(handoff_id)

for member in manifest.completed_members:
    completed[member["member_index"]] = decode_handoff_member_record(
        await client.get_handoff_member_record(member["record_ref"])
    )  # no model or tool execution

for member in manifest.checkpointed_trajectories:
    i = member["member_index"]
    if i in saved:
        completed[i] = decode_handoff_member_record(
            await client.get_handoff_member_record(saved[i]["record_ref"])
        )
        continue
    target_lease = await client.acquire_handoff_trajectory(
        handoff_id, member["trajectory_id"], "target-worker", 600.0
    )
    snapshot = await client.load_trajectory_snapshot(target_lease)
    original_kwargs = decode_handoff_member_record(
        await client.get_handoff_member_record(member["input_ref"])
    )
    # Existing ToolAgentLoop.resume_from_snapshot: keep historical token,
    # mask, logprob and completed tool results; use target_lease.generation.
    # Bind an ownership callback closing over this target_lease.
    output = await resume_existing_tool_agent(snapshot, target_lease, original_kwargs)
    await client.record_handoff_member_result(
        handoff_id, i, target_lease, encode_handoff_member_record(output), real_resume_metrics
    )
    completed[i] = output

for member in manifest.pending_samples:
    i = member["member_index"]
    if i in saved:
        completed[i] = decode_handoff_member_record(
            await client.get_handoff_member_record(saved[i]["record_ref"])
        )
        continue
    target_lease = await client.acquire_handoff_pending(
        handoff_id, member["trajectory_id"], "target-worker", 600.0
    )
    original_kwargs = decode_handoff_member_record(
        await client.get_handoff_member_record(member["record_ref"])
    )
    output = await run_existing_tool_agent_first_time(original_kwargs, target_lease)
    await client.record_handoff_member_result(
        handoff_id, i, target_lease, encode_handoff_member_record(output), real_resume_metrics
    )
    completed[i] = output
```

The two example callbacks above denote the integration owner's existing V1
generation/postprocessing functions. They are not alternative model engines.

## Publish through the existing TransferQueue

Create the exact stock V1 fields from all saved outputs and validate the original
training semantics. The binding uses the same `uid_session_id_output_index` keys
as `AgentLoopWorkerTQ`; no new consumption queue is introduced.

```python
binding = {
    "partition_id": "train",
    "keys": [f"{uid}_{i}_0" for i in range(8)],
    "content_digest": digest_of_frozen_training_fields,
}
intent = await client.prepare_publication(handoff_id, binding)
if intent["state"] == "PREPARED":
    intent = await client.mark_publication_started(handoff_id)  # before the TQ RPC
    await tq.async_kv_batch_put(keys=binding["keys"], fields=fields, tags=tags, partition_id="train")

# Read actual data using TQ's public kv_batch_get / async_kv_batch_get surface.
# Check every member, token/mask/logprob, policy source and the actual frozen
# fields digest. Do not substitute cache tags, registration or a put return code.
actual = await read_and_verify_existing_tq_keys(binding)
if actual is None:
    outcome = await client.reconcile_publication(handoff_id, None)
    # DISPATCHED / NEEDS_VERIFICATION -> BLOCKED. Do not resend the write.
else:
    receipt = {
        "publication_id": intent["publication_id"],
        "binding": binding,
        "backend_reference": actual,  # actual TQ metadata/result references
    }
    await client.reconcile_publication(handoff_id, receipt)
    await client.finish_handoff_manifest(handoff_id, intent["publication_id"])
    await client.collect_garbage()  # only after durable FINISHED
```

A fresh coordinator changes unfinished PUBLISHED records to
NEEDS_VERIFICATION. A write that was dispatched but cannot be read back remains
BLOCKED, even if an instantaneous query reports absence: an earlier RPC may
still be in flight. Queue loss, controller replay persistence and the optimizer
success/ack crash window are not repaired by this coordinator.

## Current upstream ToolAgentLoop integration

The optional subclass is a separate import so ordinary coordinator/codec users
do not load AgentLoop/model dependencies:

```python
from verl.trainer.ppo.v1.trajectory_state.preemptible_tool_agent_loop import (
    DurableToolAgentLoop, TrajectoryCheckpointYield,
)
```

Its registered name is `durable_tool_agent`. With no state controls, `run` invokes
the stock ToolAgentLoop. A source execution uses the existing hook contract:

```python
async def source_owner_is_current():
    await client.check_trajectory_owner(source_lease)

async def maybe_commit(snapshot):
    state = await client.get_trajectory_preemption(snapshot.trajectory_id)
    if state != "preempt_requested":
        return None  # Decline a checkpoint; the live agent continues in place.
    await client.mark_owned_trajectory_safe_point(source_lease)
    return await client.commit_trajectory_snapshot(source_lease, snapshot, ttl_seconds=900)

try:
    output = await source_loop.run_until_checkpoint(
        {"temperature": 0.7, "top_p": 0.9, "logprobs": True},
        commit_snapshot=maybe_commit, check_owner=source_owner_is_current,
        raw_prompt=raw_prompt, tools_kwargs=tools_kwargs,
        trajectory_id="group-0/member-2", rollout_group_id="group-0",
        rollout_sample_index=2, rollout_group_size=8,
        execution_generation=source_lease.generation,
        snapshot_policy_identity=current_actual_policy_identity,
        snapshot_environment_version="local-life-env-v1",
        snapshot_tool_version="local-life-tools-v1",
    )
except TrajectoryCheckpointYield as yielded:
    # Save original input record + yielded.reference in the group Manifest.
    # This exception is not an output to postprocess or send to TransferQueue.
    committed_snapshot_reference = yielded.reference
```

After durable Manifest commit/source release and target acquisition:

```python
snapshot = await client.load_trajectory_snapshot(target_lease)

async def target_owner_is_current():
    await client.check_trajectory_owner(target_lease)

output = await target_loop.resume_from_snapshot(
    snapshot, check_owner=target_owner_is_current,
    execution_generation=target_lease.generation,
    environment_version="local-life-env-v1", tool_version="local-life-tools-v1",
    snapshot_policy_identity=current_actual_target_policy_identity,
)
# Existing V1 output postprocessing, reward evaluation and TQ fields stay with
# the integration owner. Persist its actual completed-member record before ack.
```

The target uses stored token IDs directly; no initial prompt rendering occurs.
Tool results already present in the snapshot are never dispatched again. The
current implementation waits for the parent tool handler's complete batch and
CT append, then checks ownership before the commit callback. It does not capture
an arbitrary point during tool execution. The explicit policy identity must
match real newly generated policy evidence. The target may have new weights if
its tokenizer/template and continuation contract remain compatible; historical
policy segments retain their original versions and execution generations.

`trajectory_generation_receipts` stores every actual generate's extra fields
with its generated-token range and execution generation. Native gateway attempt
ranges remain relative to that generate, and `native_meta_info.weight_versions`
are checked before accepting it. In this first adapter a generate containing
multiple policies is refused rather than guessing the identity of older tokens.
KV references remain optional metadata; this subclass uses token continuation
and introduces no KV restore, scheduler or resource reclaim signal source.

## V1 prompt-group helper

`PromptGroupHandoffRuntime` only coordinates one prompt's existing per-member
callbacks. It does not instantiate an agent, compute rewards, pad outputs, select
replicas or call TransferQueue:

```python
from verl.trainer.ppo.v1.trajectory_state.runtime import (
    PromptGroupExecutionError, PromptGroupHandoffRuntime, PromptGroupResult,
)

runtime = PromptGroupHandoffRuntime(
    client, owner_id="worker-0", incarnation_id=actual_worker_incarnation,
    policy_identity=current_actual_policy_identity,
    environment_version="local-life-env-v1", tool_version="local-life-tools-v1",
    lease_seconds=600, snapshot_ttl_seconds=3600, preemption_deadline_s=60,
)

async def run_member(index, original_kwargs, lease, commit_snapshot, check_owner):
    # Create the current upstream loop using the ordinary worker configuration.
    loop = create_existing_agent_loop(original_kwargs["agent_name"])
    sampling = original_kwargs["sampling_params"]
    loop_kwargs = existing_agent_run_kwargs(original_kwargs)
    return await loop.run_until_checkpoint(
        sampling, commit_snapshot=commit_snapshot, check_owner=check_owner,
        **loop_kwargs,
        trajectory_id=lease.trajectory_id, rollout_group_id=uid,
        rollout_sample_index=index, rollout_group_size=len(member_kwargs),
        execution_generation=lease.generation,
        snapshot_policy_identity=runtime.policy_identity,
        snapshot_environment_version=runtime.environment_version,
        snapshot_tool_version=runtime.tool_version,
    )

result = await runtime.run_source(uid, member_kwargs, run_member)
if result.status == "completed":
    # Normal complete prompt: retain the stock V1 postprocessing/TQ path.
    outputs = result.outputs
else:
    # Returned only after durable COMMITTED and actual source rights release.
    handoff_id = result.handoff_id
```

Manager/driver control surfaces are `await runtime.request_preemption(uid)` and
`runtime.group_status(uid)`. A worker that does not own that active source returns
False/None, allowing a manager to broadcast a driver reclaim signal. A finished
source's small recent status remains queryable; an already-handed-off uid cannot
be run on that source again. `start_members=[0,1,2,3,4,5]` optionally makes the
n=8 acceptance fixture deterministic. The driver first observes members 0–1
complete and 2–5 running, then requests preemption while 6–7 have not started.
Default execution starts every member; this fixture is not a scheduling policy.

Target signatures deliberately match existing worker callback construction:

```python
async def resume_member(index, snapshot, original_kwargs, lease, check_owner):
    loop = create_existing_agent_loop(original_kwargs["agent_name"])
    return await loop.resume_from_snapshot(
        snapshot, check_owner=check_owner, execution_generation=lease.generation,
        environment_version=target_runtime.environment_version,
        tool_version=target_runtime.tool_version,
        snapshot_policy_identity=target_runtime.policy_identity,
    )

async def run_pending(index, original_kwargs, lease, check_owner):
    # Use the same original sampling/trajectory/trace/prompt, not a new sample.
    loop = create_existing_agent_loop(original_kwargs["agent_name"])
    return await loop.run(
        original_kwargs["sampling_params"], **existing_agent_run_kwargs(original_kwargs),
        _trajectory_check_owner=check_owner,
        trajectory_id=lease.trajectory_id, rollout_group_id=uid,
        rollout_sample_index=index, rollout_group_size=n,
        execution_generation=lease.generation,
        snapshot_policy_identity=target_runtime.policy_identity,
        snapshot_environment_version=target_runtime.environment_version,
        snapshot_tool_version=target_runtime.tool_version,
    )

result = await target_runtime.run_target(handoff_id, resume_member, run_pending)
# Original outputs are now ordered by member_index. Persisted target results are
# skipped on retry/reopen; only incomplete members receive a new generation.
# Existing root-owned V1 postprocessing/TQ verification follows this return.
```

The initial runtime requires identical canonical policy, tool and environment
identities on source and target. It returns `PromptGroupResult(status, outputs,
handoff_id, member_kwargs, output_record_refs, metrics)`. A handoff result has
`outputs=[]`; it cannot be consumed as a complete training group. An ordinary
completed group has no Manifest, because the existing Manifest contract requires
at least one checkpointed member. Its record refs are diagnostics, not a new
commit-before-queue exactly-once contract.

`SAFE_POINT_TIMEOUT`, callback errors and uncertain coordinator operations raise
`PromptGroupExecutionError`, with the reason, uid, possible handoff_id and any
unsettled execution rights. A snapshot committed before the group Manifest is
not advertised as a fully retained prompt group. The helper fences only its own
live rights before cancelling failed callbacks; it does not release checkpointed
source rights before Manifest COMMITTED. Unknown timing/token/resource metrics
remain null. `source_execution_release_s` measures signal-to-owner-release in the
same worker process; actual worker/resource reclaim latency remains null until a
driver observes resource reclamation. External tool exactly-once and optimizer
success-before-ack remain outside the guarantee.
