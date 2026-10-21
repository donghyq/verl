from __future__ import annotations

"""离线构造业务 ToolAgentLoop 上下文。

目标：
- 只从同一条原始 query 记录中提取 `query`、`msg_ext.lat`、`msg_ext.lng`、`user_id`、`uuid`。
- 用外部给定的原 system prompt 原样构造 `[system, user]` 两条消息。
- 形成可直接塞进 `ToolAgentLoop.run(..., tools_kwargs=...)` 的每工具 `create_kwargs`。

约束：
- 不读取其它数据源，不借用 Flywheel 或其它样本的经纬度。
- 经纬度缺失、非数值、NaN/Inf、越界时直接拒绝。
- 只透传真实 metadata，不擅自补 `level`、`poiId` 等业务参数。
- 该模块纯离线、无网络、无 tokenizer、无模型依赖。
"""

import copy
import math
from collections.abc import Iterable, Mapping
from typing import Any


class ContextBuildError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class MissingFieldError(ContextBuildError):
    pass


class InvalidCoordinateError(ContextBuildError):
    pass


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MissingFieldError("invalid_record", f"{field_name} must be a mapping object.")
    return value


def _require_non_empty_text(record: Mapping[str, Any], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise MissingFieldError("missing_required_field", f"{field_name} must be a non-empty string.")
    return value


def _require_scalar_metadata(record: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in record:
        raise MissingFieldError("missing_required_field", f"{field_name} is required.")
    value = record[field_name]
    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        raise MissingFieldError("missing_required_field", f"{field_name} must be a scalar metadata value.")
    if isinstance(value, str) and not value.strip():
        raise MissingFieldError("missing_required_field", f"{field_name} must not be blank.")
    return value


def _parse_coordinate(raw_value: Any, *, field_name: str, lower: float, upper: float) -> float:
    if isinstance(raw_value, bool):
        raise InvalidCoordinateError("invalid_coordinate", f"{field_name} must not be boolean.")
    if isinstance(raw_value, str):
        if not raw_value.strip():
            raise InvalidCoordinateError("invalid_coordinate", f"{field_name} must not be blank.")
        candidate = raw_value.strip()
    else:
        candidate = raw_value

    try:
        value = float(candidate)
    except (TypeError, ValueError) as exc:
        raise InvalidCoordinateError("invalid_coordinate", f"{field_name} must be numeric.") from exc

    if not math.isfinite(value):
        raise InvalidCoordinateError("invalid_coordinate", f"{field_name} must be finite.")
    if not lower <= value <= upper:
        raise InvalidCoordinateError(
            "invalid_coordinate",
            f"{field_name} must be within [{lower}, {upper}].",
        )
    return value


def extract_business_query_context(record: Mapping[str, Any]) -> dict[str, Any]:
    """从单条原始记录中提取严格受限的业务上下文。"""
    row = _require_mapping(record, field_name="record")
    query = _require_non_empty_text(row, "query")
    user_id = _require_scalar_metadata(row, "user_id")
    uuid = _require_scalar_metadata(row, "uuid")

    msg_ext = _require_mapping(row.get("msg_ext"), field_name="msg_ext")
    lat = _parse_coordinate(msg_ext.get("lat"), field_name="msg_ext.lat", lower=-90.0, upper=90.0)
    lng = _parse_coordinate(msg_ext.get("lng"), field_name="msg_ext.lng", lower=-180.0, upper=180.0)

    return {
        "query": query,
        "msg_ext": {
            "lat": lat,
            "lng": lng,
        },
        "user_id": user_id,
        "uuid": uuid,
    }


def build_prompt_messages(*, system_prompt: str, query: str) -> list[dict[str, str]]:
    """按 `[system, user]` 固定结构重建消息，system prompt 不做任何改写。"""
    if not isinstance(system_prompt, str) or system_prompt == "":
        raise ContextBuildError("invalid_system_prompt", "system_prompt must be a non-empty string.")
    if not isinstance(query, str) or not query.strip():
        raise ContextBuildError("missing_required_field", "query must be a non-empty string.")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]


def build_tool_create_kwargs(record: Mapping[str, Any], *, system_prompt: str) -> dict[str, Any]:
    """为单个工具实例构造 `BaseTool.create(create_kwargs=...)` 所需的上下文。"""
    business_query_context = extract_business_query_context(record)
    messages = build_prompt_messages(system_prompt=system_prompt, query=business_query_context["query"])
    return {
        "messages": messages,
        "business_query_context": business_query_context,
    }


def build_tools_kwargs(
    record: Mapping[str, Any],
    *,
    system_prompt: str,
    tool_names: Iterable[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """构造 `ToolAgentLoop.run(..., tools_kwargs=...)` 需要的每工具入参。

    返回形状：
    {
        "tool_name": {
            "create_kwargs": {
                "messages": [...],
                "business_query_context": {...},
            }
        }
    }

    注意：
    - 这里只生成 `create_kwargs`，不引入任何模型应自行决定的业务参数。
    - 当前 `ToolAgentLoop` 只会把 `create_kwargs` 传给 `BaseTool.create(...)`；
      `FunctionTool` 明确忽略 `tools_kwargs`。
    """
    create_kwargs = build_tool_create_kwargs(record, system_prompt=system_prompt)
    result: dict[str, dict[str, dict[str, Any]]] = {}

    for raw_name in tool_names:
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ContextBuildError("invalid_tool_name", "tool_names must contain non-empty strings only.")
        tool_name = raw_name.strip()
        if tool_name in result:
            raise ContextBuildError("duplicate_tool_name", f"Duplicate tool name is not allowed: {tool_name}")
        result[tool_name] = {
            "create_kwargs": copy.deepcopy(create_kwargs),
        }
    return result
