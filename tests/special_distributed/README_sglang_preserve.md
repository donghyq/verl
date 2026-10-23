This is an opt-in GPU correctness candidate, not an executed GPU result.

Apply the preserved-weight-update implementation and this new-file patch to
the fixed VeRL review base. Use Linux x86_64, Python 3.12 and the repository's
unchanged `uv.lock`. Its SGLang path selects SGLang 0.5.20, sglang-kernel 0.4.7,
Torch 2.13.0+cu130, Transformers 5.12.1 and Ray 2.55.1. Do not substitute a
different SGLang source checkout or the older CUDA 12.9 environment.

```bash
uv sync --frozen --extra fsdp --extra sglang --extra test
uv run --no-sync python examples/sglang_preserve_requests.py --gate-only
CUDA_VISIBLE_DEVICES='' uv run --no-sync python -c \
  'import torch; from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter; from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica; from verl.checkpoint_engine import CheckpointEngineManager; assert not torch.cuda.is_initialized(); print("FULL_MODULE_IMPORT_PASS_NO_CUDA")'
OMP_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 1200s \
  uv run --no-sync python examples/sglang_preserve_requests.py \
  --run-gpu --rollout-tp 2 --output /tmp/sglang-crossweight-fresh
```

The last command requires **three dedicated visible GPUs**: one real SGD/NCCL
sender and two native SGLang TP workers. The result directory must not exist.
TP1 (`--rollout-tp 1`) uses two GPUs and is a separate debugging scope.
The repository's `docker/Dockerfile.uv.cu130` is a matching environment recipe;
no prebuilt image or complete Linux cache has been verified by this preparation.

The equivalent opt-in test is:

```bash
OMP_NUM_THREADS=1 VERL_RUN_SGLANG_PRESERVE_GPU=1 timeout 1200s \
  uv run --no-sync pytest -q \
  tests/special_distributed/test_sglang_preserve_requests.py
```

The probe reuses the existing tiny Qwen3 MoE and real SGD/NCCL sender. It calls
the actual `CheckpointEngineManager` and `ServerAdapter`, pauses with native
`in_place`, transfers complete parameters with native rank-specific CUDA IPC
loading, and resumes the same unfinished RID. It checks both TP workers:

- Native **loaded** router parameter digests match the sender before and after
  the real SGD update, and change. Full sender parameter names/bytes and native
  successful tensor-loader calls are recorded; this is not a byte comparison of
  every fused native expert parameter against the HF parameter layout.
- The request, committed KV bytes/storage, token/logprob prefix and MoE routing
  stay intact while paused. Successful native forward processing records actual
  input positions and output intervals. Newly launched forward inputs must have
  zero overlap with the historical prefix; pending old overlap results are not
  miscounted as newly executed forwards.
- Every native tensor bucket reports success with `flush_cache=False`,
  `abort_all_requests=False` and both TP entries. Radix and overlap remain enabled.
- A new request uses the new version's cache salt and observes zero old-prefix
  cache hits.

`evidence.json` preserves raw per-rank observations. `result.json` reports PASS
only after every assertion; errors preserve `first-error.txt` and owned cleanup
evidence. Missing routing/forward evidence fails rather than fabricating a count.
SG 0.5.20's tensor request does not supply a native weight-version span here;
that field remains UNKNOWN. VeRL entry/exit steps and per-forward parameter
digests are separate evidence. An already launched overlap forward may commit
after resume using the old parameters; its actual interval remains labelled.

This is a tiny-MoE smoke test for a mixed-weight in-flight request. It does not
prove that its suffix equals a fresh request recomputed entirely with the new
weights, nor cover GDN/Mamba, speculative decoding, DP/PP, LoRA, quantization,
cross-host movement, business throughput or Trainer integration.

## Recorded validation and remaining checks

The functional portion of a Linux/A100 TP2 run at commit `3fdcd5226` passed:
one real SGD update, 10 preserved output tokens/logprobs/routing entries and
502 further outputs. Both native ranks recorded 501 outputs using the updated
router parameters and zero historical input recomputation. The native overlap
scheduler completed one additional input after the request finished; it produced
no output and remains included in the compute count. The same environment ran
84 public CPU tests with no skips, failures, errors, or CUDA initialization.

That command returned **nonzero** after writing the functional evidence: the
immediate post-shutdown child-process check rejected four PIDs. Their individual
states were not recorded. The outer owned-process guard subsequently recorded
zero remaining tracked descendants. Do not describe this as a complete CLI pass.
The cleanup follow-up waits for live children and records zombie states using the
native shutdown resource contract. Four actual CPU child-process lifecycle cases
pass; the revised cleanup has not been revalidated with a native GPU engine.

The GPU run used a separate frozen dependency snapshot with three explicit,
retained dependency-audit exceptions (NumPy/mistral-common, absent flash-attn-4,
and the cuSPARSELt package version). It does not validate the repository's
unmodified default lock, the documented Docker recipe, or full upstream CI.
Those checks and human line-by-line review remain pending. AI assistance was used.
