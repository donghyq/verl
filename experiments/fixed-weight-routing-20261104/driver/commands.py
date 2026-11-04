"""Only CPU-executed exact commands may enter the GPU screen."""
from pathlib import Path


def variants(base):
    if base[1] != "--host" or "launch" in base or "--disable-retries" not in base:
        raise ValueError("expected validated root CLI with retries disabled")
    out = {}
    for name in ("native_default", "native_tuned", "native_session"):
        command = list(base)
        if name == "native_tuned":
            command += ["--balance-abs-threshold", "0", "--balance-rel-threshold", "1.0"]
        elif name == "native_session":
            command[command.index("--policy") + 1] = "manual"
            command += ["--assignment-mode", "min_load"]
        out[name] = command
    return out


def checked_command(gate, arm, binary, model):
    if gate.get("status") != "PASS_THREE_EXACT_CPU_GATEWAY_COMMANDS":
        raise ValueError("actual CPU CLI execution is required")
    command = list(gate["commands"][arm])
    if command[0] != str(binary) or command[command.index("--model-path")+1] != str(model):
        raise ValueError("model or binary differs from CPU command")
    if "--disable-retries" not in command or "launch" in command:
        raise ValueError("unsafe or unvalidated command")
    mapping = {"30615": "30515", "30616": "30516", "http://127.0.0.1:30613": "http://127.0.0.1:30513",
               "http://127.0.0.1:30614": "http://127.0.0.1:30514"}
    return [mapping.get(token, token) for token in command]
