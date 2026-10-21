# Project B trajectory state migration

## Fixed sources

- Community integration base: `acad9875a8bdfc81afbcb0a50d146b2630f44093`.
- Existing M1 protocols: `f91c201d52665b835a92e119b013f36ec9a49c58`.
- `source_provenance.json` lists original files, symbols, line ranges and exact
  source/destination SHA256 values. Only local import namespaces changed for
  migrated codecs; the JSON schema, validation and private member format stay
  unchanged.

| Existing code | V1 module | Treatment |
|---|---|---|
| `CanonicalPolicyKVIdentity`, canonical fingerprint and identity decoder | `identity.py` | Exact contract extraction; no legacy Trainer dependency |
| Snapshot/owner dataclasses, canonical bytes and token digest | `snapshot.py` | Exact codec, import namespace changed |
| Complete native KV object receipt validation | `kv_export_proof.py` | Exact proof; optional reference only, no new transport |
| Cooperative preemption states and transitions | `handoff.py` | Verbatim existing state machine |
| Manifest dataclass, commit validation and states | `handoff.py` | Exact codec and registry; coordinator restricts one prompt group |
| Member cloudpickle/base64/digest envelope | `handoff.py` | Exact private runtime codec |
| `register_producer`, current generation selection | `generation.py` | Verbatim methods with injected durable mapping; no new epoch |
| Snapshot registry coordination | `coordinator.py` | Persistent pointers, original history checks and generation rules |
| Actor-resident member records and target progress | `coordinator.py` | Immutable records before persistent references |
| Full-group Artifact publication | Existing V1 TransferQueue | Stable keys and content verification; no copied queue or Trainer |
| Existing ToolAgentLoop checkpoint/resume entry contract | `preemptible_tool_agent_loop.py` | Registered opt-in subclass; invokes current upstream handlers and CT builder |

## Required small changes

1. The state record includes both existing producer histories: source Rollouter
   generation and per-trajectory execution generation. Old schema-2 Artifact
   snapshots contain neither the trajectory table nor M1 manifests/progress.
2. Snapshot bytes and their full readback precede one durable latest-reference
   and SNAPSHOT_COMMITTED update. The separate marker RPC is idempotent.
3. Original member inputs, completed outputs and target-finished outputs have
   opaque references backed by verified immutable bytes. Identical bytes may
   share physical storage, but distinct logical owners cannot delete each
   other's not-yet-attached records.
4. Source rights are released only after COMMITTED is durable. Target generation
   allocation is durable before the new lease is returned.
5. Old execution leases are never hydrated on restart. Old monotonic times are
   not used for ownership or retention on a different node. Timing metrics from
   a restarted coordinator are null; the driver must measure its own intervals.
6. A queue write intent and DISPATCHED state precede the stock TQ call. Reading
   actual fields under the stable keys is necessary to confirm publication.
   Unknown outcomes become BLOCKED; the module never blindly retries the write.
7. FINISHED and the verified queue reference are durable before GC. Cleanup is
   an explicit, retryable operation over recorded objects only.
8. New snapshots retain the original fourteen AgentData fields and codec. The
   `extra_fields` records the actual current Continuous Token implementation,
   constant boundary/config values and selected tool schema. This rejects old
   pre-CT snapshots whose continuation semantics cannot be proven compatible.
   Text builders have no hidden per-request cursor; multimodal processor state,
   custom unaudited builders and routed-expert replay are rejected.
9. The subclass preserves committed token IDs, masks and behavior logprobs, and
   resumes directly at GENERATING. It delegates all prompt append, tool execution
   and parsing to upstream methods. It checks the resulting generated-token
   history after CT merges; unsafe removal of generated history fails closed.
10. A ContextVar observes actual TokenOutput results without monkeypatching a
    shared server. Actual native response ranges/weight versions (or a covering
    legacy single-version receipt) define provenance. One generate call with
    multiple policies is currently rejected. Historical turns may have distinct
    real policies; they are never stamped with the latest training step.

## Scope and operational assumptions

- One active writer per run. The filesystem must correctly implement cross-client
  advisory flock, atomic same-directory rename, hard links and fsync. A lifetime
  lock rejects a second coordinator. This is not a multi-writer consensus service.
- The default immutable semantic-state store is the same shared filesystem.
  `ObjectStore` can be supplied only if its successful puts and readbacks satisfy
  the intended fault domain. In-memory Mooncake survival is not disk durability.
- The default state survives coordinator-process failure while the shared
  filesystem remains available. It does not claim simultaneous loss of storage,
  the TQ controller, the trainer or the entire training job.
- Every Manifest contains one text-agent prompt group and one final TQ output per
  member. Arbitrary mid-token checkpoints, multi-group transactions and external
  tool exactly-once transactions are out of scope.
- TQ writes, its ReplayBuffer indices and optimizer updates remain stock V1. An
  optimizer-success-before-ack crash is not made exactly-once by these records.
- The adapters in `API_EXAMPLES.md` describe V1 wiring points. Shared AgentLoop,
  TrainerBase, checkpoint and engine entry points are owned by the integration
  owner; this workstream does not modify them.

## Verification

New tests cover coordinator kill/reopen, both sides of the atomic metadata
replace, stale owner rejection, generation monotonicity, retained member
results, corrupt state/object rejection, a second writer, publication uncertainty
and FINISHED-before-GC. Source hashes document the protocol migration separately.

CPU tests and source compilation must run on CodeLab 617728841. No local tests or
models were executed while preparing this workstream. Test results are pending.

The ToolAgentLoop tests additionally use actual upstream Continuous Token and
FunctionTool handlers with a fixture tokenizer and fake model TokenOutput. They
exercise in-flight tool waiting, owner expiry, immutable resume history, zero
committed-tool replay, concurrent request context isolation, policy provenance,
and incompatible continuation rejection. They do not establish model, storage
performance, optimizer or controlled-cluster acceptance.

`runtime.py` supplies the opt-in V1 prompt-group callback helper. It preserves
original per-member kwargs using the existing member codec, records completed
outputs, stops not-yet-started members after a driver signal, and commits/releases
one group Manifest. Target execution skips durable restored outcomes and reads
the original completed outputs without generating them again. The helper never
publishes TQ or implements reward/Trainer logic. Normal completed groups retain
the stock V1 output path and do not fabricate an all-completed HandoffManifest.
Committed-Manifest recovery and ordinary completion-before-publication are
different guarantees. Source and target policy identities are fixed in this
initial integration; no new policy epoch is introduced.

Production failure cleanup calls `abort_owned_trajectory(lease)` once. The
serialized coordinator validates ownership, advances a current source tracker
to ABORTED and persists that execution's release atomically. It never relies on
a previous check followed by an unconditional ID-only mutation. A committed
source snapshot is not aborted, and a failed target lease is released without
rewriting the source's HANDED_OFF history. The original ID-only abort API remains
an explicit control/admin surface, not a runtime cleanup operation.

Production registration, reclaim requests and safe-point transitions also use
owned variants with the original lease. Each validates that actual execution
lease inside the same actor transaction before advancing the existing tracker.
The original ID-only methods remain administration/test controls. A valid old
read followed by coordinator restart/new-generation takeover cannot authorize
any of the three later control writes. Snapshot/Manifest schemas and generation
history are unchanged.
