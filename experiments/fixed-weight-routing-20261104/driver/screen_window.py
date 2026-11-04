"""Own two TP1 processes and one Gateway within one registered GPU window."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

WORK = Path(__file__).resolve().parent
PORTS = (30513, 30514)
GATEWAY_PORT = 30515
BINARY = Path.home() / "project-b-gateway-cache-salt-20261104-r1/binary/smg"
from commands import checked_command

def validate_binary_gate(gate, binary):
    if hashlib.sha256(binary.read_bytes()).hexdigest() != gate["binary_sha256"]:
        raise ValueError("CPU-validated binary changed")


def save(name, value):
    tmp = WORK / (name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(WORK / name)


def validate(window, now, work):
    if (window.get("status") != "APPROVED" or window.get("approval_source") != "user_message"
            or not window.get("approval_turn_id") or window.get("codelab_id") != "617728841"
            or window.get("work_dir") != str(work) or type(window.get("gpu_count")) is not int
            or window.get("gpu_count") != 2 or window.get("scope") != "FIXED_WEIGHT_ROUTING_COMPONENT_SCREEN"
            or window.get("keep_instance_after_window") is not True
            or window.get("actual_device_name") != "NVIDIA A800-SXM4-80GB"):
        raise ValueError("resource/scope differs from registered bridge")
    start, deadline = window.get("started_at"), window.get("absolute_deadline")
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in [start, deadline])
            or not start <= now < deadline or not 0 < deadline - start <= 2400):
        raise ValueError("invalid or expired bridge window")
    uuids = window.get("gpu_uuids")
    if not isinstance(uuids, list) or len(uuids) != 2 or len(set(uuids)) != 2 or any(not x.startswith("GPU-") for x in uuids):
        raise ValueError("two distinct registered physical GPUs required")
    return deadline


def matches(row, cmdline, pgid, ticks):
    argv = cmdline.split(b"\0")
    return (row["pgid"] == pgid and row["start_ticks"] == ticks and row["identity_argument"].encode() in argv
            and str(row["port"]).encode() in argv)


def cleanup():
    errors = []
    for name in ["gateway-owned.json", "engine-0-owned.json", "engine-1-owned.json"]:
        path = WORK / name
        if not path.exists():
            continue
        row = json.loads(path.read_text()); proc = Path("/proc") / str(row["pid"])
        try:
            ticks = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
            if not matches(row, (proc / "cmdline").read_bytes(), os.getpgid(row["pid"]), ticks):
                raise RuntimeError("process identity changed; refusing to signal " + name)
            os.killpg(row["pgid"], signal.SIGTERM)
            for _ in range(20):
                if not proc.exists():
                    break
                time.sleep(.25)
            else:
                os.killpg(row["pgid"], signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError):
            pass
        except Exception as exc:
            errors.append(str(exc))
    return errors


def spawn(name, command, env, port, identity):
    log = (WORK / (name + ".log")).open("ab")
    proc = subprocess.Popen(command, env=env, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ticks = Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    row = {"pid": proc.pid, "pgid": os.getpgid(proc.pid), "start_ticks": ticks, "port": port,
           "identity_argument": str(identity), "command": command, "started": time.time()}
    save(name + "-owned.json", row)
    return proc, log


def launcher(window, deadline):
    inventory = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=True, timeout=5).stdout.splitlines()
    rows = [line.split(",") for line in inventory]
    if (len(rows) != 2 or {r[0].strip() for r in rows} != set(window["gpu_uuids"])
            or any(r[1].strip() != window["actual_device_name"] or int(r[2]) != 81920 for r in rows)):
        raise ValueError("GPU identity differs from registered A800 pair")
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    if apps:
        raise ValueError("another GPU compute process exists; no launch")
    save("physical-gpu-inventory.json", {"checked_at": time.time(), "rows": inventory})
    for port in [*PORTS, GATEWAY_PORT, 30516]:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError("port already owned: " + str(port))
    gate = json.loads((WORK / "cpu-cli-gate.json").read_text())
    validate_binary_gate(gate, BINARY)
    for name, expected in json.loads((WORK / "source-manifest.json").read_text()).items():
        if hashlib.sha256((WORK / name).read_bytes()).hexdigest() != expected:
            raise ValueError("registered source changed: " + name)
    runtime = Path.home() / "project-b-microbench-20261103"
    model = Path.home() / "models/Qwen3.5-35B-A3B-BF16"
    gateway_commands = [checked_command(gate, arm, BINARY, model) for arm in ("native_default", "native_tuned", "native_session")]
    env = dict(os.environ, **json.loads((runtime / "runtime-probe.json").read_text())["environment"])
    env.pop("PYTHONPATH", None); env.pop("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", None)
    env.update(SGLANG_ENABLE_TOKEN_KV_SNAPSHOT="1", SGLANG_ENABLE_NATIVE_CACHE_DIRECTORY="1",
               SGLANG_ENABLE_STABLE_NATIVE_TARGET="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
    processes = []
    try:
        for index, port in enumerate(PORTS):
            # Retain existing compiled caches; do not claim cold-start performance.
            cache = Path.home() / "project-b-dual-replica-capacity-20261103-r1" / f"cache-{index}"
            per_env = dict(env, CUDA_VISIBLE_DEVICES=window["gpu_uuids"][index],
                TRITON_CACHE_DIR=str(cache / "triton"), TORCH_EXTENSIONS_DIR=str(cache / "torch"),
                FLASHINFER_CACHE_DIR=str(cache / "flashinfer"),
                PROJECT_B_PREFILL_CHECKPOINT_POLICY="1")
            command = [str(runtime / "env/bin/python"), str(WORK / "candidate_entry.py"), "--model-path", str(model),
                "--tp", "1", "--port", str(port), "--weight-version", "1", "--dtype", "bfloat16",
                "--mem-fraction-static", "0.94", "--context-length", "17408", "--max-running-requests", "4",
                "--disable-cuda-graph", "--disable-overlap-schedule", "--attention-backend", "triton",
                "--linear-attn-backend", "triton", "--enable-deterministic-inference",
                "--mamba-radix-cache-strategy", "extra_buffer", "--mamba-ssm-dtype", "float32", "--mamba-track-interval", "64"]
            processes.append(spawn(f"engine-{index}", command, per_env, port, WORK / "candidate_entry.py"))
        def wait_health(proc, port, label):
            while time.time() < deadline - 60:
                if any(p.poll() is not None for p, _ in processes):
                    raise RuntimeError("owned process exited; diagnose original log before retry")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                        if response.status == 200:
                            save(label + "-ready.json", {"pid": proc.pid, "ready_at": time.time()})
                            return
                except Exception:
                    pass
                time.sleep(1)
            raise TimeoutError("startup reached absolute cleanup boundary")
        for index, port in enumerate(PORTS):
            wait_health(processes[index][0], port, f"engine-{index}")
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/server_info", timeout=2) as response:
                info = json.loads(response.read())
            save(f"engine-{index}-server-info.json", info)
            if (info.get("tp_size") != 1 or info.get("dtype") != "bfloat16" or info.get("context_length") != 17408
                    or info.get("max_running_requests") != 4
                    or info.get("enable_deterministic_inference") is not True
                    or info.get("mamba_radix_cache_strategy") != "extra_buffer"
                    or info.get("mamba_ssm_dtype") != "float32" or Path(info.get("model_path", "")).name != model.name):
                raise ValueError("actual engine configuration differs from registered same-model TP1")
        with (WORK / "screen-driver.log").open("ab") as log:
            probe = subprocess.run([str(runtime / "env/bin/python"), str(WORK / "screen_driver.py")],
                    cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(.1, deadline - 50 - time.time()))
        probe.check_returncode()
    finally:
        errors = cleanup()
        for proc, log in processes:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                errors.append("owned parent has not exited: " + str(proc.pid))
            log.close()
        save("launcher-cleanup.json", {"finished": time.time(), "errors": errors})
        if errors:
            raise RuntimeError("owned cleanup incomplete: " + str(errors))


def main():
    window = json.loads((WORK / "registered-window.json").read_text())
    deadline = validate(window, time.time(), WORK)
    if hashlib.sha256((WORK / "source-manifest.json").read_bytes()).hexdigest() != window["source_manifest_sha256"]:
        raise ValueError("registered source manifest changed")
    if "--launcher" in sys.argv:
        launcher(window, deadline)
        return
    started = time.time()
    with (WORK / "supervisor-once.json").open("x") as file:
        json.dump({"pid": os.getpid(), "started": started}, file)
    with (WORK / "launcher.log").open("ab") as log:
        child = subprocess.Popen([sys.executable, "-u", str(WORK / "screen_window.py"), "--launcher"],
                                 cwd=WORK, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save("launcher-owned.json", {"pid": child.pid, "pgid": os.getpgid(child.pid), "started": time.time()})
    error = None
    try:
        while child.poll() is None:
            if time.time() >= deadline - 40:
                raise TimeoutError("absolute UTC cleanup boundary")
            time.sleep(.5)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:1500]}
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=2)
        cleanup_errors = cleanup()
        occupancy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=5)
        if occupancy.returncode or occupancy.stdout.strip():
            cleanup_errors.append("post-cleanup GPU processes remain; inspect ownership")
        save("screen-terminal.json", {"status": "COMPLETED_COMPONENT_SCREEN_NO_RL_CLAIM" if child.returncode == 0 and not error and not cleanup_errors else "FAILED_DIAGNOSE_BEFORE_RETRY",
             "started": started, "finished": time.time(), "absolute_deadline": deadline,
             "returncode": child.returncode, "error": error, "cleanup_errors": cleanup_errors,
             "post_cleanup_compute_processes": occupancy.stdout.splitlines(),
             "keep_instance_after_window": True, "performance_or_rl_gain": False, "model_loading_and_window_seconds": time.time()-started, "window_proxy_gpu_hours": (time.time()-started)*2/3600})


if __name__ == "__main__":
    main()
