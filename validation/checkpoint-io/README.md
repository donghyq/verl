# Checkpoint writer diagnostic and optional candidate

Run the small CPU probe against an owned writable backing directory and a local
temporary directory:

```bash
CUDA_VISIBLE_DEVICES='' python checkpoint_write_probe_three_arm.py OWN_BACKING_DIR LOCAL_TMP_DIR
```

The three arms save identical 33 MiB CPU state: a filename destination, a
16 MiB buffered Python file, and a local save followed by a 16 MiB chunked copy.
The reported write interval includes creation, flush and fsync; readback/value
validation and cleanup have separate timings. Only this probe's UUID files are removed.

`buffered-native-save.patch` is an opt-in experiment for the existing four-rank
checkpoint guard, not a general checkpoint backend or an upstream feature patch.
Its baseline guard SHA is
`bf615f998c73a3121cd000d739458060735ad6917c0a8a55d266d1f66358ee3e`.
With `VERL_P1_BUFFERED_NATIVE_SAVE=1`, only the current native save context's
model, optimizer and extra-state file for that rank uses the buffered file object.
Original objects, arguments, paths, exceptions and COMPLETE requirements remain.
The candidate retains the original flush/close behavior and adds no fsync; do not
infer crash durability from a native save return. Other saves retain their path.

Local Torch 2.11 checks cover real small model/Adam/scheduler/RNG readback, a
matching subsequent CPU update, aliases/metadata and error propagation without
retry. They do not establish target Torch 2.13 compatibility, large sharded-state
peak memory, checkpoint throughput or GPU savings. The patch remains disabled
until target-environment I/O and full-state validation justify enabling it.

AI assistance was used. Human review, target measurements and deployment are pending.


## Large-storage preflight

`large_storage_cpu_preflight.py --execute OWN_EXISTING_HDD_DIR` compares the original filename writer with a 16 MiB buffered file, using a 256 MiB CPU backing storage shared by model tensors and views. It verifies alias layout, model/Adam/scheduler/Python+NumPy+Torch RNG restoration and the next actual CPU optimizer update. Each arm runs in a fresh process, records save/fsync and peak RSS, and removes its own UUID file. Default invocation does not run the probe.

The deployment guard requires Linux and the original Torch 2.13.0+cu129 runtime. A development smoke check of the same function body on macOS/Torch 2.11 passed both arms after replacing only the runtime assertion in the test loader; the candidate file was unchanged. This is functional development evidence, not target-runtime I/O performance or distributed FSDP validation.

The companion policy defines a prospective four-worker admission bound before loading the large model. All four workers must pass the state checks and buffered save bound. The 40 MiB/s floor is a planning threshold based on an estimated model-plus-two-Adam-state volume; it is not a measured 35B throughput guarantee or a training-benefit result. Target-runtime evidence is recorded below. AI assistance was used; the source fallback and review corrections are disclosed in the project records.


## Target-runtime result

The first completed target-worker preflight used Torch 2.13.0+cu129 and the exact 256 MiB probe above. Filename saving took 47.62 s (5.38 MiB/s); the 16 MiB buffered writer took 48.36 s (5.29 MiB/s), including flush/fsync/close. Both preserved the CPU state, alias layout and next optimizer update. Buffering did not improve the observed write rate and failed the fixed 40 MiB/s planning gate.

Only one worker produced a complete CPU record; the other three records are UNKNOWN. The training application stopped before large-model loading, so this result provides no FSDP/full-checkpoint recovery or training-benefit claim. The next control separates local serialization from 16 MiB chunked copying to the target path and records process CPU time. See `observed-target-runtime-io.json`; no real business input or internal host/path is included.
