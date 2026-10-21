"""Load the pinned original reward runtime without replacing community modules."""
from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
import hashlib
import json
import math
import numbers
from enum import IntEnum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


ORIGINAL_COMMIT = "1d754ea1078cb29ef01fab230beb4b0eed84a16c"

_AGENT_LOOP_RELATIVE_PATH = "verl/experimental/agent_loop/agent_loop.py"
_RL_FACTORY_AGENT_LOOP_RELATIVE_PATH = "verl/experimental/agent_loop/rl_factory_agent_loop.py"
_API_CALL_RELATIVE_PATH = "verl/utils/reward_score/deepsearch/api_call.py"
_JUDGE_API_CLIENT_RELATIVE_PATH = "verl/utils/reward_score/judge_api_client.py"

_EXPECTED_SHA256 = {
    _AGENT_LOOP_RELATIVE_PATH: "49b615301b8f34db302e0d446b07fbac0ae22658789b50c6259f03db1a16d1ab",
    _API_CALL_RELATIVE_PATH: "4bf27c76f4087c1a857f70d52461ce05d53b90e55aa491fe2437afa23437a4c6",
    _RL_FACTORY_AGENT_LOOP_RELATIVE_PATH: "49e54088f2bb357400e72f83ce146771e11f3eaeec8e70966691ec289166c364",
    _JUDGE_API_CLIENT_RELATIVE_PATH: "9919969ef9e52f985248bece021c95e213d33b65bbbe203c99c210b61f07f12d",
}


class OriginalRewardSourceError(RuntimeError):
    """The pinned original reward source is unavailable or invalid."""


class SourceHashMismatchError(OriginalRewardSourceError):
    """A pinned original reward source file does not match its required hash."""


@dataclass(frozen=True)
class OriginalRewardRuntime:
    compute_score_server: Callable[..., Any]
    agent_error_type: type
    build_assistant_message: Callable[..., Any]
    receipt: dict


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_source(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SourceHashMismatchError(f"cannot read pinned source file: {path}") from exc


def _verify_source_files(source_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path, expected_hash in _EXPECTED_SHA256.items():
        path = source_root / relative_path
        if not path.is_file():
            raise SourceHashMismatchError(f"pinned source file is missing: {relative_path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise SourceHashMismatchError(
                f"pinned source hash mismatch for {relative_path}: expected {expected_hash}, got {actual_hash}"
            )
        hashes[relative_path] = actual_hash
    return hashes


def _extract_agent_error(tree: ast.Module, filename: str) -> ast.ClassDef:
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentError"]
    if len(matches) != 1:
        raise OriginalRewardSourceError(f"expected exactly one AgentError class in {filename}")
    return matches[0]


def _extract_build_assistant_message(tree: ast.Module, filename: str) -> ast.FunctionDef:
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef) or class_node.name != "RLFactoryToolAgentLoop":
            continue
        matches = [
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_build_assistant_message"
        ]
        if len(matches) != 1:
            raise OriginalRewardSourceError(
                f"expected exactly one _build_assistant_message method in {filename}"
            )
        method = matches[0]
        if method.decorator_list:
            raise OriginalRewardSourceError("_build_assistant_message must not have decorators")
        return method
    raise OriginalRewardSourceError(f"RLFactoryToolAgentLoop was not found in {filename}")


def _make_agent_error_proxy(agent_error_type: type) -> ModuleType:
    proxy = ModuleType(_AGENT_LOOP_RELATIVE_PATH.removesuffix(".py").replace("/", "."))
    proxy.AgentError = agent_error_type
    return proxy


def _make_private_import(agent_error_proxy: ModuleType, api_call_module: ModuleType | None = None):
    default_import = builtins.__import__
    agent_loop_name = _AGENT_LOOP_RELATIVE_PATH.removesuffix(".py").replace("/", ".")
    api_call_name = _API_CALL_RELATIVE_PATH.removesuffix(".py").replace("/", ".")

    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name == agent_loop_name and fromlist == ("AgentError",):
            return agent_error_proxy
        if (
            api_call_module is not None
            and level == 0
            and name == api_call_name
            and fromlist == ("async_reward_server_call",)
        ):
            return api_call_module
        return default_import(name, globals, locals, fromlist, level)

    return private_import


def _execute_isolated_module(
    source: str,
    filename: Path,
    module_name: str,
    private_import,
) -> ModuleType:
    module = ModuleType(module_name)
    module.__file__ = str(filename)
    module.__dict__["__builtins__"] = {**builtins.__dict__, "__import__": private_import}
    code = compile(source, str(filename), "exec")
    exec(code, module.__dict__)
    return module


def _load_agent_error(agent_loop_source: str, agent_loop_path: Path) -> type:
    tree = ast.parse(agent_loop_source, filename=str(agent_loop_path))
    agent_error_ast = _extract_agent_error(tree, str(agent_loop_path))
    namespace = {
        "IntEnum": IntEnum,
        "__name__": "_project_b_original_reward_agent_error",
    }
    module_ast = ast.Module(body=[agent_error_ast], type_ignores=[])
    exec(compile(module_ast, str(agent_loop_path), "exec"), namespace)
    agent_error_type = namespace.get("AgentError")
    if not isinstance(agent_error_type, type):
        raise OriginalRewardSourceError("the extracted AgentError is not a class")
    return agent_error_type


def _load_build_assistant_message(
    rl_factory_source: str,
    rl_factory_path: Path,
) -> Callable[..., Any]:
    tree = ast.parse(rl_factory_source, filename=str(rl_factory_path))
    method_ast = _extract_build_assistant_message(tree, str(rl_factory_path))
    if not method_ast.args.args or method_ast.args.args[0].arg != "self":
        raise OriginalRewardSourceError("_build_assistant_message must start with self")
    method_ast.args.args = method_ast.args.args[1:]
    future_annotations = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations")],
        level=0,
    )
    ast.copy_location(future_annotations, method_ast)
    module_ast = ast.Module(body=[future_annotations, method_ast], type_ignores=[])
    ast.fix_missing_locations(module_ast)

    module = ModuleType("_project_b_original_reward_build_assistant_message")
    module.__file__ = str(rl_factory_path)
    module.__dict__["json"] = json
    module.__dict__["__builtins__"] = builtins.__dict__.copy()
    exec(compile(module_ast, str(rl_factory_path), "exec"), module.__dict__)
    return module._build_assistant_message


def load_original_reward_runtime(source_root) -> OriginalRewardRuntime:
    """Load the pinned original reward functions in isolated namespaces."""
    root = Path(source_root)
    source_hashes = _verify_source_files(root)

    agent_loop_path = root / _AGENT_LOOP_RELATIVE_PATH
    api_call_path = root / _API_CALL_RELATIVE_PATH
    judge_api_client_path = root / _JUDGE_API_CLIENT_RELATIVE_PATH
    rl_factory_agent_loop_path = root / _RL_FACTORY_AGENT_LOOP_RELATIVE_PATH

    agent_loop_source = _read_source(agent_loop_path)
    api_call_source = _read_source(api_call_path)
    judge_api_client_source = _read_source(judge_api_client_path)
    rl_factory_agent_loop_source = _read_source(rl_factory_agent_loop_path)

    agent_error_type = _load_agent_error(agent_loop_source, agent_loop_path)
    agent_error_proxy = _make_agent_error_proxy(agent_error_type)

    api_call_private_import = _make_private_import(agent_error_proxy)
    api_call_module = _execute_isolated_module(
        api_call_source,
        api_call_path,
        "_project_b_original_reward_api_call",
        api_call_private_import,
    )

    judge_private_import = _make_private_import(agent_error_proxy, api_call_module)
    judge_api_client_module = _execute_isolated_module(
        judge_api_client_source,
        judge_api_client_path,
        "_project_b_original_reward_judge_api_client",
        judge_private_import,
    )

    build_assistant_message = _load_build_assistant_message(
        rl_factory_agent_loop_source,
        rl_factory_agent_loop_path,
    )

    receipt = {
        "original_commit": ORIGINAL_COMMIT,
        "source_files_sha256": source_hashes,
        "loading_method": {
            "AgentError": "AST-extracted original class",
            "api_call.py": "full isolated module execution",
            "judge_api_client.py": "full isolated module execution",
            "_build_assistant_message": "AST-extracted original method",
        },
        "isolation": {
            "original_reward_modules_registered_in_sys_modules": False,
            "global_builtins_replaced": False,
            "public_verl_modules_replaced": False,
        },
    }

    return OriginalRewardRuntime(
        compute_score_server=judge_api_client_module.compute_score_server,
        agent_error_type=agent_error_type,
        build_assistant_message=build_assistant_message,
        receipt=receipt,
    )


def validate_reward_messages(extra_info, *, require_final_answer=True) -> list[dict]:
    """Validate the original reward wrapper and return its message list.

    Invalid inputs raise ``ValueError``. The pinned source scorer also accepts
    the observed history at its turn/length limit. That history must stay
    unchanged; it is not a naturally completed answer.
    """
    if not isinstance(extra_info, dict):
        raise ValueError("extra_info must be a dict")
    if "messages" not in extra_info:
        raise ValueError("extra_info['messages'] is missing")

    wrapper = extra_info["messages"]
    if not isinstance(wrapper, dict) or set(wrapper) != {"messages"}:
        raise ValueError("extra_info['messages'] must be exactly {'messages': [...]}")

    messages = wrapper["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("extra_info['messages']['messages'] must be a non-empty list")

    allowed_roles = {"system", "user", "assistant", "tool"}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be a dict")
        if message.get("role") not in allowed_roles:
            raise ValueError(f"message {index} has an invalid role")

    if require_final_answer:
        last_message = messages[-1]
        if last_message.get("role") != "assistant":
            raise ValueError("the last message must be an assistant message")
        if last_message.get("tool_calls"):
            raise ValueError("the last assistant message must not contain unfinished tool calls")

    return messages


def validate_reward_result(result, agent_error_type) -> dict:
    """Validate a real reward result without recalculating or changing its score.

    Invalid results raise ``ValueError``.
    """
    if not isinstance(result, dict):
        raise ValueError("reward result must be a dict")
    if "score" not in result or "error" not in result:
        raise ValueError("reward result must contain score and error")

    score = result["score"]
    try:
        finite = math.isfinite(score)
    except (TypeError, ValueError, OverflowError):
        finite = False
    if isinstance(score, bool) or not isinstance(score, numbers.Real) or not finite:
        raise ValueError("reward score must be a finite number and must not be a bool")

    error = result["error"]
    if isinstance(error, bool) or not isinstance(error, (agent_error_type, int)):
        raise ValueError("reward error must be the original AgentError or an integer")
    error_code = int(error)
    if error_code != int(agent_error_type.Success):
        raise ValueError("reward error must be AgentError.Success")

    return {
        "score": score,
        "error_code": error_code,
        "reward_valid": True,
    }


def validate_reward_service_response(response) -> dict:
    """Reject nonempty service errors that the original client defaults to zero."""
    if isinstance(response, str):
        response = json.loads(response)
    if not isinstance(response, dict) or not isinstance(response.get('reward'), dict):
        raise ValueError('reward service response has no reward object')
    score = response['reward'].get('total')
    if isinstance(score, bool) or not isinstance(score, numbers.Real) or not math.isfinite(score):
        raise ValueError('reward service response has no finite numeric total')
    return response
