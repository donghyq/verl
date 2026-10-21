"""Run exactly AA + ABBA using the original owned inference supervisor.

Execution is explicit and Linux/MLP-only. Import and offline assessment do not
load a model, start Ray, contact a service or allocate a GPU.
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from benefit_metrics import ORDER, assess, window_metrics
from benefit_native import load_module, save, sha


def inventory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total",
        "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=10, check=True)
    rows = [[x.strip() for x in row] for row in csv.reader(result.stdout.splitlines())]
    if (len(rows) != 2 or any(len(row) != 5 or "H20" not in row[2] for row in rows)
            or len({row[1] for row in rows}) != 2):
        raise ValueError("This isolated MLP job requires exactly two assigned H20 devices")
    return rows


def assert_allocation(spec, rows):
    binding = spec["allocation"]
    if (os.environ.get("AFO_APPID") != binding["app_id"]
            or Path("/proc/sys/kernel/random/boot_id").read_text().strip() != binding["boot_id"]
            or [row[1] for row in rows] != binding["actual_gpu_uuids"]):
        raise ValueError("Actual app/kernel/GPU allocation changed")
    # Fresh independent job only. Neither previous CodeLab UUIDs nor process
    # names authorize GPU reuse or signals against existing computations.
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, timeout=10, check=True)
    if result.stdout.strip():
        raise ValueError("Assigned GPUs already have compute processes; do not touch them")
    if any(int(row[4])-int(row[3]) < 70000 for row in rows):
        raise ValueError("Original 70GiB-per-device free headroom guard failed")


def owner(spec, root):
    """Bind the unchanged R6 process owner to this job's runtime and allocation."""
    sys.path.insert(0, spec["original_process_directory"])
    module = load_module("closeout_original_engine_owner", spec["original_supervisor"])
    module.CAP = Path(spec["owner_capsule"])
    module.PRIVATE = Path(spec["sglang_source"])
    module.ROOT = Path(root)
    module.PY = spec["python"]
    module.ENV = Path(spec["owner_environment"])
    from benefit_model_stage import verify_for_engine
    module.MODEL = verify_for_engine(spec)
    original_popen = module.subprocess.Popen
    profile = spec["engine_profile"]
    from benefit_startup import memory_snapshot,owned_process_snapshot
    process_module=load_module("closeout_owner_process_evidence",Path(spec["original_process_directory"])/"portable_process.py")
    active={"process":None,"identity":None,"phase_root":None,"last_observed":0.}
    def observe_owned(force=False):
        process=active["process"]
        if process is None or (not force and time.monotonic()-active["last_observed"]<15):return
        if active["identity"] is None:return
        actual=process_module.identity(process.pid)
        if actual is not None and not process_module.same_process(actual,active["identity"]):return
        members=process_module.group_members(process.pid)
        members=[row for row in members if all(row.get(key)==active["identity"].get(key)
            for key in ("boot_id","pid_namespace_inode","session_id"))]
        row={"event":"owned_engine_startup_or_phase_observed","utc_unix_s":time.time(),
            "memory":memory_snapshot(),"processes":owned_process_snapshot(members),"launcher_returncode":process.poll(),
            "prior_diagnostic_error":active.get("diagnostic_error")}
        with (active["phase_root"]/"owner-observations.jsonl").open("a") as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        active["last_observed"]=time.monotonic()
    def launch(argv, *args, **kwargs):
        # Only frozen configuration values differ from the historical 16K
        # component window. Original model/tool generation budgets are intact.
        if not isinstance(argv, list) or argv[1:3] != ["-m", "sglang.launch_server"]:
            raise ValueError("Original supervisor may launch only its native SGLang server")
        if verify_for_engine(spec) != module.MODEL:
            raise ValueError("Source and target must share the exact committed local checkpoint")
        expected = {"--context-length": "16384", "--weight-version": "0"}
        for key, value in expected.items():
            index = argv.index(key)
            if argv[index+1] != value: raise ValueError("Original supervisor argv drifted")
        argv = list(argv)
        argv[argv.index("--context-length")+1] = str(profile["context_length"])
        for forbidden in ("--weight-loader-disable-mmap", "--weight-loader-prefetch-checkpoints",
                          "--model-loader-extra-config", "--load-format", "--cpu-offload-gb"):
            if forbidden in argv:
                raise ValueError("Original owner loading argv drifted: " + forbidden)
        argv.extend(["--load-format", "safetensors", "--model-loader-extra-config",
                     '{"enable_multithread_load":false}', "--cpu-offload-gb", "0"])
        # The pinned native defaults retain mmap. Refuse environment changes
        # that select a different, unbudgeted weight loader implementation.
        for key in ("SGLANG_ENABLE_WEIGHT_LOADER_V2", "SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D"):
            if kwargs["env"].get(key, "0").lower() not in ("0", "false"):
                raise ValueError("Unbudgeted native host-memory loading option: " + key)
        nccl_root=Path(spec["owner_environment"]).parent/"nccl-info"
        nccl_root.mkdir(exist_ok=True)
        kwargs["env"]=dict(kwargs["env"],NCCL_DEBUG="INFO",
                           NCCL_DEBUG_FILE=str(nccl_root/"nccl.%h.%p.log"))
        process=original_popen(argv, *args, **kwargs)
        directory=Path(kwargs["stdout"].name).parent
        active.update(process=process,identity=None,phase_root=directory,last_observed=0.)
        environment=kwargs["env"]
        # Return the Popen handle even if a diagnostic write fails. The original
        # owner must receive/register it before any later exception can unwind.
        try:
            active["identity"]=process_module.identity(process.pid)
            sources={"original_supervisor":Path(spec["original_supervisor"]),
                     "original_process_owner":Path(spec["original_process_directory"])/"portable_process.py",
                     "actual_snapshot_manager":Path(spec["sglang_source"])/"sglang/srt/managers/token_kv_snapshot.py",
                     "benefit_owner_adapter":Path(__file__)}
            source_identity={}
            for role,path in sources.items():
                try:source_identity[role]={"path":str(path),"sha256":sha(path)}
                except OSError as error:source_identity[role]={"path":str(path),"sha256":None,"reason":str(error)}
            save(directory/"engine-launch-provenance.json",{"actual_argv":argv,"identity":active["identity"],
                "source_modules":source_identity,
                "inherited_runtime_environment":{key:environment.get(key) for key in
                    ("PATH","PYTHONPATH","LD_PRELOAD","LD_LIBRARY_PATH","VIRTUAL_ENV","CUDA_VISIBLE_DEVICES",
                     "NCCL_DEBUG","NCCL_DEBUG_FILE","NCCL_DEBUG_SUBSYS")},
                "memory_before_model_load":memory_snapshot(),"model_path_same_for_both_phases":module.MODEL,
                "original_model_path":spec["model_path"],"local_model_stage_receipt":spec["model_stage_receipt"],
                "original_readiness_seconds":300,"window_seconds":spec["window_seconds"]})
            observe_owned()
        except Exception as error:
            active["diagnostic_error"]=type(error).__name__+": "+str(error)
        return process
    module.subprocess = SimpleNamespace(Popen=launch, run=subprocess.run, PIPE=subprocess.PIPE,
        STDOUT=subprocess.STDOUT, DEVNULL=subprocess.DEVNULL)
    def actual_inventory():
        rows = inventory(); assert_allocation(spec, rows)
        return [[row[0], row[1], row[3], row[4]] for row in rows]
    module.inventory = actual_inventory
    def observed_sleep(seconds):
        observe_owned()
        return time.sleep(seconds)
    module.time=SimpleNamespace(time=time.time,sleep=observed_sleep)
    def interrupted(signum, frame):
        raise TimeoutError("Owned benchmark supervisor interrupted at the frozen deadline")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:module.main()
    finally:observe_owned(force=True)


def _read(path): return json.loads(Path(path).read_text())


def _cleanup_proven(path):
    if not path.is_file(): return False
    row = _read(path)
    return row.get("actual_process_exit_observed") is True and row.get("source_process_group_empty") is True


def run_window(spec, spec_path, root, arm, number):
    root.mkdir(parents=True, exist_ok=False)
    started, monotonic = time.time(), time.monotonic()
    command = [sys.executable, str(Path(__file__).resolve()), "engine-owner", "--spec", str(spec_path), "--output", str(root)]
    process_module = load_module("closeout_original_process", Path(spec["original_process_directory"])/"portable_process.py")
    children = []; streams = []
    owner_stream = (root/"supervisor.log").open("wb", buffering=0); streams.append(owner_stream)
    process = subprocess.Popen(command, stdout=owner_stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True)
    owner_identity = process_module.identity(process.pid)
    save(root/"supervisor-process.json", {"identity": owner_identity, "argv": command})
    launched = set()
    error = None
    try:
        while process.poll() is None:
            if time.monotonic()-monotonic >= spec["window_seconds"]:
                raise TimeoutError("Finite 1200-second window elapsed; no retry/extension")
            for phase in ("source", "target"):
                path = root/phase/"ready.json"
                if phase in launched or not path.is_file(): continue
                ready = _read(path)
                if not ready["url"].startswith("http://127.0.0.1:"):
                    raise ValueError("Benchmark can contact only its owned loopback engine")
                args = [sys.executable, str(Path(__file__).resolve()), "phase", "--spec", str(spec_path),
                    "--output", str(root), "--name", phase, "--url", ready["url"], "--arm", arm]
                stream = (root/phase/"phase-driver.log").open("wb", buffering=0); streams.append(stream)
                child = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=""), stdin=subprocess.DEVNULL, start_new_session=True)
                child_identity = process_module.identity(child.pid)
                children.append((child, child_identity)); launched.add(phase)
                save(root/phase/"phase-driver-process.json", {"identity": child_identity, "argv": args})
            time.sleep(.25)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        # CPU phase children are also exclusively owned. Stop them before
        # asking the original owner to clean its engine process group.
        for child, identity in children:
            if child.poll() is None:
                process_module.stop_source(child, identity, root/("cpu-child-exit-%s.json"%child.pid),
                                           "benchmark-cpu-phase", timeout_s=20)
        if process.poll() is None:
            if not process_module.same_process(process_module.identity(process.pid), owner_identity):
                raise RuntimeError("Original supervisor PID identity changed; no signal sent")
            process.send_signal(signal.SIGTERM)
        try: process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            error = {"type": "CleanupNotVerified", "message": "Original owned supervisor did not finish cleanup; no next window"}
        for stream in streams: stream.close()
    elapsed = time.monotonic()-monotonic
    terminal = _read(root/"terminal.json") if (root/"terminal.json").is_file() else {}
    good = (error is None and terminal.get("status")=="BOTH_ENGINE_PHASES_COMPLETED_PENDING_INDEPENDENT_ACCEPTANCE"
            and _cleanup_proven(root/"source-exit.json") and _cleanup_proven(root/"target-exit.json"))
    groups = _read(root/"target-groups.json") if (root/"target-groups.json").is_file() else []
    fixture = _read(root/"fixture.json") if (root/"fixture.json").is_file() else {}
    counters = [_read(root/phase/"model-counters.json") for phase in ("source", "target")
                if (root/phase/"model-counters.json").is_file()]
    dispatches = _read(root/"source-groups.json") if (root/"source-groups.json").is_file() else []
    business_seconds = (max(x["first_dispatch_monotonic"]+g["completion_seconds"] for x,g in zip(dispatches,groups))
                        - min(x["first_dispatch_monotonic"] for x in dispatches)) if groups and dispatches else None
    row = {"arm": arm, "window_index": number, "status": "COMPLETE" if good else "INCOMPLETE",
        "actual_gpu_uuids": spec["allocation"]["actual_gpu_uuids"], "groups": groups,
        "started_unix_s": started, "ended_unix_s": time.time(), "raw_owned_window_seconds": elapsed,
        "total_allocated_seconds": elapsed, "request_business_seconds": business_seconds,
        "actual_new_model_tokens": sum(x["actual_new_model_tokens"] for x in counters),
        "source_and_target_cleanup_verified": good, "error": error, "original_owner_terminal": terminal,
        "failed_attempts_are_not_retried": True, "failure_distribution": {} if good else {"window_failed": 1},
        "worker_internal_p95": None, "worker_internal_p95_reason": "Only same-kernel driver/coordinator observations are available",
        "group_p95_scope": "First original group dispatch through native full-group persistence and original reward; same kernel monotonic clock",
        "source_and_target_both_cold_started": True, "source_only_prefix_warm_in_both_arms": True,
        "cross_host_hot_migration": False, "optimizer_updates": 0,
        **{key:fixture.get(key) for key in ("fixture_sha256", "sampling_sha256", "policy_identity_sha256",
                                         "original_reward_implementation_sha256")}}
    if good:
        try: window_metrics(row)
        except ValueError as exc: row.update(status="INCOMPLETE", error={"type":"NativeAcceptanceIncomplete", "message":str(exc)})
    from recovery_cost_hooks import summarize
    try:
        save(root/"recovery-net-cost.json", summarize(root, row, allocation=spec.get("actual_allocation_interval")))
    except Exception as observation_error:
        print("RECOVERY_NET_COST_UNAVAILABLE:"+type(observation_error).__name__,flush=True)
    save(root/"window.json", row)
    return row


def campaign(spec, spec_path, root):
    if sys.platform != "linux" or not os.environ.get("AFO_APPID"):
        raise RuntimeError("Owned execution requires the independent ROOT-submitted MLP allocation")
    if spec["window_seconds"] != 1200 or spec["maximum_windows"] != 6:
        raise ValueError("Frozen AA+ABBA budget differs")
    rows = []; started = spec["allocation"]["assigned_start_monotonic"]
    def interrupted(signum, frame):
        raise TimeoutError("Original absolute work deadline interrupted the finite campaign")
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        for index, arm in enumerate(ORDER):
            remaining = spec["absolute_deadline_unix_s"]-time.time()-spec["cleanup_reserve_seconds"]
            if remaining < spec["window_seconds"]:
                raise TimeoutError("Not enough original assigned budget for another whole window")
            row = run_window(spec, spec_path, root/("window-%d-%s"%(index,arm)), arm, index)
            rows.append(row); save(root/"windows.json", rows)
            if row["status"] != "COMPLETE": break
            if index == 1:
                prior = [window_metrics(x) for x in rows]
                save(root/"prior-noise.json", {"completed_before_comparison": True, "same_arm": "token_recompute",
                    "windows": prior, "extra_windows_allowed": False})
    except BaseException as error:
        save(root/"campaign-error.json", {"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()})
    finally:
        total = time.monotonic()-started
        # Keep raw windows and the entire real allocation. Amortize common
        # staging/prepare/gaps equally only when the complete fixed campaign ran.
        shared = total-sum(x["raw_owned_window_seconds"] for x in rows)
        if len(rows)==6 and all(x["status"]=="COMPLETE" for x in rows):
            for row in rows:
                row["common_allocation_cost_share_seconds"] = shared/6
                row["total_allocated_seconds"] = row["raw_owned_window_seconds"]+shared/6
        result = assess(rows)
        if not rows:
            result.update(status="NOT_RUN",reason="ABSOLUTE_BUDGET_DID_NOT_ADMIT_ONE_COMPLETE_WINDOW")
        result.update(actual_total_assigned_gpu_hours=2*total/3600, actual_total_assigned_seconds=total,
            actual_gpu_uuids=spec["allocation"]["actual_gpu_uuids"], common_staging_prepare_gap_seconds=shared,
            shared_cost_amortized_only_after_all_six_windows_complete=True,
            independent_job_topology="one worker, two H20, source exits before target starts",
            maximum_windows=6, execution_attempts=len(rows), optimizer_updates=0)
        save(root/"windows.json", rows); save(root/"terminal.json", result)
        print(json.dumps({"event":"BENEFIT_TERMINAL", "status":result["status"],
            "windows":len(rows), "gpu_hours":result["actual_total_assigned_gpu_hours"]}), flush=True)
    return 0 if result["status"]=="OBSERVED_FINITE_BUSINESS_BENEFIT" else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("engine-owner","phase","run","prepare","assess"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--name", choices=("source","target"));parser.add_argument("--url")
    parser.add_argument("--arm", choices=set(ORDER))
    args=parser.parse_args()
    if args.mode=="assess":
        save(args.output,assess(_read(args.windows)));return 0
    spec=_read(args.spec)
    if args.mode=="engine-owner": owner(spec,args.output);return 0
    if args.mode=="phase":
        from benefit_native import phase
        phase(spec,args.output,args.name,args.url,args.arm);return 0
    if args.mode=="prepare":
        from benefit_native import Context
        context=Context(spec,args.output)
        fixture,_=asyncio.run(context.fixture())
        return 0 if fixture["status"]=="PREPARED_NOT_EXECUTED" else 2
    return campaign(spec,args.spec,args.output)


if __name__=="__main__": raise SystemExit(main())
