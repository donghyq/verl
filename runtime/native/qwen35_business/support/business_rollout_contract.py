"""Strict, source-pinned rollout budgets for the bounded validation driver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any


_REQUIRED_FIELDS = {
    "prompt_length",
    "response_length",
    "max_model_len",
    "max_assistant_turns",
    "max_user_turns",
    "max_tool_response_length",
    "max_new_tokens",
    "temperature",
    "top_p",
    "tool_timeout",
    "max_retries",
    "sampling_seed",
    "source_sha256",
    "proof_scope",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_ASSISTANT_TURNS = 6


@dataclass(frozen=True)
class RolloutContract:
    """One validated rollout budget and its safe receipt fields."""

    prompt_length: int
    response_length: int
    max_model_len: int
    max_assistant_turns: int
    max_user_turns: int | None
    max_tool_response_length: int
    max_new_tokens: int
    temperature: float
    top_p: float | None
    tool_timeout: float
    max_retries: int
    sampling_seed: int
    source_sha256: str | None
    proof_scope: str
    mode: str = "explicit_rollout_contract"

    def rollout_config(self) -> dict[str, Any]:
        """Return the config consumed by ToolAgentLoop and partial rollout."""

        return {
            "actor_rollout_ref": {
                "rollout": {
                    "name": "sglang",
                    "full_determinism": False,
                    "prompt_length": self.prompt_length,
                    "response_length": self.response_length,
                    "multi_turn": {
                        "format": "business_function_xml",
                        "max_user_turns": self.max_user_turns,
                        "max_assistant_turns": self.max_assistant_turns,
                        "max_parallel_calls": 1,
                        "max_tool_response_length": self.max_tool_response_length,
                        "tool_response_truncate_side": "right",
                    },
                }
            }
        }

    def gateway_config(
        self,
        *,
        base_url: str,
        expected_worker_urls: tuple[str, ...],
        request_timeout_s: float = 240.0,
        admission_timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Return the native gateway fields derived from this contract."""

        return {
            "base_url": base_url,
            "max_model_len": self.max_model_len,
            "prompt_length": self.prompt_length,
            "response_length": self.response_length,
            "expected_worker_urls": expected_worker_urls,
            "request_timeout_s": request_timeout_s,
            "admission_timeout_s": admission_timeout_s,
        }

    def sampling_params(self) -> dict[str, Any]:
        """Return sampling fields without exposing prompts or credentials."""

        params: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "logprobs": True,
            "sampling_seed": self.sampling_seed,
        }
        if self.top_p is not None:
            params["top_p"] = self.top_p
        return params

    def manager_settings(self) -> dict[str, Any]:
        """Return the original manager's timeout and retry fields."""

        return {
            "max_turns": self.max_assistant_turns,
            "max_retries": self.max_retries,
            "tool_timeout": self.tool_timeout,
        }

    def receipt(self) -> dict[str, Any]:
        """Return a receipt containing only budgets and non-secret metadata."""

        result = asdict(self)
        result["prompt_plus_response_tokens"] = self.prompt_length + self.response_length
        result["budget_relation_checked"] = self.mode == "explicit_rollout_contract"
        result["production_replay"] = False
        result["credentials_in_receipt"] = False
        result["applied_to"] = [
            "ToolAgentLoop",
            "PartialRollout through FullyAsyncLLMServerClient",
            "GatewayClientConfig",
            "sampling_params",
            "original manager timeout and retry",
        ]
        return result


def _checked_integer(
    payload: Mapping[str, Any], name: str, *, minimum: int, allow_none: bool = False
) -> int | None:
    value = payload[name]
    if allow_none and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"rollout contract field {name} must be an integer")
    if value < minimum:
        raise ValueError(f"rollout contract field {name} must be at least {minimum}")
    return value


def _checked_number(
    payload: Mapping[str, Any],
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"rollout contract field {name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < minimum or (exclusive_minimum and value == minimum):
        raise ValueError(f"rollout contract field {name} is outside its valid range")
    if maximum is not None and value > maximum:
        raise ValueError(f"rollout contract field {name} is outside its valid range")
    return value


def _checked_sha256(payload: Mapping[str, Any]) -> str:
    value = payload["source_sha256"]
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("rollout contract field source_sha256 must be a 64-character lowercase SHA-256")
    return value


def _checked_proof_scope(payload: Mapping[str, Any]) -> str:
    value = payload["proof_scope"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("rollout contract field proof_scope must be a non-empty string")
    if len(value) > 512 or any(ord(char) < 32 for char in value):
        raise ValueError("rollout contract field proof_scope contains unsafe formatting")
    return value


def parse_rollout_contract(payload: Mapping[str, Any]) -> RolloutContract:
    """Validate an explicit mapping without filling in defaults."""

    if not isinstance(payload, Mapping):
        raise ValueError("rollout contract must be a JSON object")
    missing = sorted(_REQUIRED_FIELDS - set(payload))
    if missing:
        raise ValueError(f"rollout contract is missing fields: {', '.join(missing)}")
    unknown = sorted(set(payload) - _REQUIRED_FIELDS)
    if unknown:
        raise ValueError(f"rollout contract has unknown fields: {', '.join(unknown)}")

    prompt_length = _checked_integer(payload, "prompt_length", minimum=1)
    response_length = _checked_integer(payload, "response_length", minimum=1)
    max_model_len = _checked_integer(payload, "max_model_len", minimum=1)
    max_assistant_turns = _checked_integer(payload, "max_assistant_turns", minimum=1)
    max_user_turns = _checked_integer(payload, "max_user_turns", minimum=1, allow_none=True)
    max_tool_response_length = _checked_integer(payload, "max_tool_response_length", minimum=1)
    max_new_tokens = _checked_integer(payload, "max_new_tokens", minimum=1)
    max_retries = _checked_integer(payload, "max_retries", minimum=0)
    sampling_seed = _checked_integer(payload, "sampling_seed", minimum=0)
    temperature = _checked_number(payload, "temperature", minimum=0.0)
    top_p = _checked_number(payload, "top_p", minimum=0.0, maximum=1.0, exclusive_minimum=True)
    tool_timeout = _checked_number(payload, "tool_timeout", minimum=0.0, exclusive_minimum=True)
    if prompt_length + response_length > max_model_len:
        raise ValueError("rollout contract prompt_length plus response_length exceeds max_model_len")
    if max_new_tokens > response_length:
        raise ValueError("rollout contract max_new_tokens exceeds response_length")
    if max_tool_response_length > response_length:
        raise ValueError("rollout contract max_tool_response_length exceeds response_length")
    if max_assistant_turns > _MAX_ASSISTANT_TURNS:
        raise ValueError("rollout contract cannot expand max_assistant_turns beyond 6")

    return RolloutContract(
        prompt_length=prompt_length,
        response_length=response_length,
        max_model_len=max_model_len,
        max_assistant_turns=max_assistant_turns,
        max_user_turns=max_user_turns,
        max_tool_response_length=max_tool_response_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tool_timeout=tool_timeout,
        max_retries=max_retries,
        sampling_seed=sampling_seed,
        source_sha256=_checked_sha256(payload),
        proof_scope=_checked_proof_scope(payload),
    )


def load_rollout_contract(path: str | Path) -> RolloutContract:
    """Read and validate one explicit JSON contract; never execute code."""

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"rollout contract has duplicate field: {key}")
            result[key] = value
        return result

    contract_path = Path(path)
    with contract_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream, object_pairs_hook=reject_duplicate_keys)
    return parse_rollout_contract(payload)


def legacy_driver_contract() -> RolloutContract:
    """Return the exact old bounded-driver values when no contract is supplied."""

    return RolloutContract(
        prompt_length=12288,
        response_length=32768,
        max_model_len=16384,
        max_assistant_turns=6,
        max_user_turns=None,
        max_tool_response_length=8192,
        max_new_tokens=4096,
        temperature=0.0,
        top_p=None,
        tool_timeout=30.0,
        max_retries=3,
        sampling_seed=42,
        source_sha256=None,
        proof_scope="legacy bounded validation driver; not a production launch replay",
        mode="legacy_driver",
    )
