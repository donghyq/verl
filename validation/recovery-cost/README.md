# Recovery-cost validation candidates

These are CPU-reviewed candidates, not a measured training speedup.
The current first-request pair uses the same weight version and a fixed saved
boundary. It is not the trained-version, cross-weight production benchmark.

The request builder retains tokens, logprobs, completed tool results, sampling
parameters and remaining generation budget. The producer patch records actual
completed native forward work for the ordinary recompute arm and the original
response body for physical recovery.

Apply the first-generation binding and then the first-request-ordering patch to
the existing recovery-cost candidate. The selected member's first A/B request
must finish before other members can enter the target semaphore. Cancellation
or failure propagates without retrying an uncertain request. Original runtime
leases, owner checks, native identity/digest checks and complete business outputs
remain required. Empty counters or serial concurrency do not prove a fresh engine.

```bash
python3 -B test_first_request_ordering.py
```

The 20 checks execute the actual new gate with CPU callback/snapshot fixtures.
They do not acquire live leases, call an HTTP engine, run a model or use GPUs.
The preceding call-point candidate had 23 separate CPU checks; those are from
its previous revision and are not combined into a new runtime pass.

Before GPU execution, bind dedicated owned fresh A/B targets, regenerate strict
source/archive identities, and use a newly committed snapshot with a valid lease.
Then compare actual work, end conditions and quality, including source export,
write/read/import and total resource holding costs. The current patch records
request-fragment observations and does not itself establish whole-window benefit.

The checkpoint I/O probe in `../checkpoint-io/` compares path-based Torch saving,
16 MiB Python buffering, and local saving followed by a 16 MiB chunked copy.
It uses 33 MiB of CPU tensor data and removes only its own UUID files. Write
timings include creation, flush and fsync; load/value validation is timed separately.
Local Torch 2.11 checks cover format, values and cleanup only. Target-environment
I/O, full FSDP checkpoints and production gains still require separate measurements.

AI assistance was used. Human line-by-line review and production validation are
pending. The feature review Draft is https://github.com/donghyq/verl/pull/1 .
