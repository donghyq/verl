from __future__ import annotations

"""Project B 业务 ToolAgentLoop 的 BaseTool 包装器。

边界：
- 只包装已存在的同步 `manager.call_tool` 可调用对象，不重写技能解析或伪造服务。
- 只在当前 trajectory 的 `agent_data._active_tools/_active_tool_schemas` 上做渐进工具更新。
- 不修改 `ToolAgentLoop` 主线，不改共享 registry，不扩展 Durable snapshot 协议。
"""

import asyncio
import copy
import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator, SchemaError

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

try:  # pragma: no cover - package import when this directory is treated as a package
    from .business_query_context import extract_business_query_context
except ImportError:  # pragma: no cover - direct import when loaded from file path/sys.path
    from business_query_context import extract_business_query_context


StatsDict = dict[str, Any]
ToolRegistry = Mapping[str, Any]


class BusinessToolAdapter(BaseTool):
    """将同步业务 tool manager 适配到 `BaseTool` 生命周期。"""

    _STATS_FIELD = "business_tool_adapter_stats"
    _CONTEXT_KEYS = {"userId", "uuid", "lat", "lng", "originalQuery"}

    def __init__(
        self,
        config: dict[str, Any],
        tool_schema: OpenAIFunctionToolSchema,
        *,
        manager_call_tool: Callable[..., str],
        full_tool_registry: ToolRegistry,
        fixed_tool_kwargs: Mapping[str, Any] | None = None,
        backend_parameter_schema: Mapping[str, Any] | None = None,
    ) -> None:
        self._manager_call_tool = manager_call_tool
        self._full_tool_registry = dict(full_tool_registry)
        self._fixed_tool_kwargs = dict(fixed_tool_kwargs or {})
        self._backend_parameter_schema = copy.deepcopy(backend_parameter_schema)
        if self._CONTEXT_KEYS.intersection(self._fixed_tool_kwargs):
            raise ValueError("Per-query context must come from the current record, not fixed tool configuration.")
        self._instance_lock = threading.RLock()
        self._instance_contexts: dict[str, dict[str, Any]] = {}
        super().__init__(config=config, tool_schema=tool_schema)

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        if not isinstance(create_kwargs, Mapping):
            raise ValueError("create_kwargs must be a mapping.")

        context = extract_business_query_context(create_kwargs.get("business_query_context", {}))
        messages = create_kwargs.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError("create_kwargs.messages must be a list if provided.")

        resolved_instance_id = instance_id or str(uuid4())
        payload = {
            "business_query_context": copy.deepcopy(context),
            "messages": copy.deepcopy(messages),
        }
        with self._instance_lock:
            if resolved_instance_id in self._instance_contexts:
                raise ValueError("Tool instance context already exists.")
            self._instance_contexts[resolved_instance_id] = payload
        return resolved_instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be a dict.")

        with self._instance_lock:
            context_payload = copy.deepcopy(self._instance_contexts.get(instance_id))

        if context_payload is None:
            stats = self._build_stats(status="missing_context", text="")
            self._record_stats(kwargs.get("agent_data"), stats)
            return ToolResponse(text=f"Business tool '{self.name}' has no active instance context."), 0.0, stats

        contract_error = self._parameter_contract_error(parameters)
        if contract_error is not None:
            stats = self._build_stats(status="invalid_parameters", text=contract_error)
            self._record_stats(kwargs.get("agent_data"), stats)
            return ToolResponse(text=contract_error), 0.0, stats

        tool_kwargs = self._build_tool_kwargs(
            parameters=parameters,
            business_query_context=context_payload["business_query_context"],
        )

        try:
            raw_result = await asyncio.to_thread(
                self._manager_call_tool,
                self.name,
                copy.deepcopy(parameters),
                tool_kwargs=tool_kwargs,
            )
        except Exception:
            stats = self._build_stats(status="backend_exception", text="")
            self._record_stats(kwargs.get("agent_data"), stats)
            return (
                ToolResponse(text=f"Business tool '{self.name}' failed in the injected backend manager."),
                0.0,
                stats,
            )

        if not isinstance(raw_result, str):
            stats = self._build_stats(status="backend_protocol_error", text="")
            self._record_stats(kwargs.get("agent_data"), stats)
            return ToolResponse(text="Business tool backend returned an invalid result type."), 0.0, stats
        result_text = raw_result
        status = self._classify_result_status(result_text)

        if self.name == "load_skill":
            try:
                refresh_status = self._maybe_refresh_tools(result_text=result_text, agent_data=kwargs.get("agent_data"))
            except (TypeError, ValueError):
                refresh_status = "refresh_rejected"
            if refresh_status is not None:
                status = refresh_status

        stats = self._build_stats(status=status, text=result_text)
        self._record_stats(kwargs.get("agent_data"), stats)
        return ToolResponse(text=result_text), 0.0, stats

    def _parameter_contract_error(self, parameters: Mapping[str, Any]) -> str | None:
        schema = self._backend_parameter_schema
        if schema is None:
            schema = self.tool_schema.model_dump(exclude_unset=True, exclude_none=True)["function"]["parameters"]
        schema = copy.deepcopy(schema)
        for prop in schema.get("properties", {}).values():
            # This alias is explicit in the original business input_fields.
            # It describes the same JSON value as array<string>; do not coerce
            # a model-produced string or rewrite the original schema view.
            declared_type = prop.get("type")
            if isinstance(declared_type, str) and declared_type in {"list[string]", "List<String>"}:
                prop["type"] = "array"
                prop.setdefault("items", {"type": "string"})
            elif isinstance(declared_type, str) and declared_type in {"String", "Integer", "Double", "Boolean"}:
                prop["type"] = {"String":"string", "Integer":"integer", "Double":"number", "Boolean":"boolean"}[declared_type]
        try:
            Draft202012Validator.check_schema(schema)
            error = next(Draft202012Validator(schema).iter_errors(parameters), None)
        except SchemaError:
            return f"工具 {self.name} 的参数定义无法验证，未调用后端。"
        if error is None:
            return None
        field = str(next(iter(error.absolute_path), "parameters"))
        if error.validator == "required":
            field = next((name for name in error.validator_value if name not in parameters), field)
        if error.validator == "type" and error.validator_value == "array":
            return (
                f"工具 {self.name} 的参数 {field} 必须是 JSON 数组，不能直接写文本或数字。"
                "请按原参数定义使用数组格式，并保留原查询含义。后端尚未执行。"
            )
        return f"工具 {self.name} 的参数 {field} 不符合原定义（{error.validator}），后端尚未执行。"

    async def release(self, instance_id: str, **kwargs) -> None:
        del kwargs
        with self._instance_lock:
            self._instance_contexts.pop(instance_id, None)

    def _build_tool_kwargs(self, *, parameters: Mapping[str, Any], business_query_context: Mapping[str, Any]) -> dict[str, Any]:
        # The original manager merges tool_kwargs after model parameters.
        # Remove overlapping defaults here so that merge cannot change model arguments.
        passthrough = {key: copy.deepcopy(value) for key, value in self._fixed_tool_kwargs.items() if key not in parameters}
        sample_context_values = {
            "userId": self._string_identifier(business_query_context["user_id"]),
            "uuid": self._string_identifier(business_query_context["uuid"]),
            "lat": business_query_context["msg_ext"]["lat"],
            "lng": business_query_context["msg_ext"]["lng"],
            "originalQuery": business_query_context["query"],
        }
        for key, value in sample_context_values.items():
            if key in parameters:
                continue
            passthrough.setdefault(key, value)
        return passthrough

    @staticmethod
    def _string_identifier(value: Any) -> str:
        # The actual API declares String, while source query records can carry
        # integer IDs. Decimal serialization is lossless; never borrow an ID,
        # accept a floating-point approximation, or generate a replacement.
        if isinstance(value, str) and value.strip():
            return value
        if type(value) is int:
            return str(value)
        raise ValueError("Original business identifier cannot be serialized as an API string.")

    def _maybe_refresh_tools(self, *, result_text: str, agent_data: Any) -> str | None:
        if agent_data is None:
            return None

        payload = self._parse_json_object(result_text)
        if payload is None:
            return None
        if payload.get("error"):
            return "tool_error"

        refresh = payload.get("should_refresh_tools")
        if refresh is False or refresh is None:
            return None
        if refresh is not True:
            return "refresh_rejected"
        if not isinstance(payload.get("content"), str) or not payload["content"].strip():
            return "refresh_rejected"
        new_tools = payload.get("new_tools")
        if not isinstance(new_tools, list):
            return "refresh_rejected"

        current_active = getattr(agent_data, "_active_tools", None)
        if not isinstance(current_active, Mapping):
            current_active = {self.name: self}
        candidate_active = dict(current_active)
        candidate_order = list(candidate_active.keys())

        validated_names: list[str] = []
        for raw_schema in new_tools:
            declared_schema = self._normalize_schema(raw_schema)
            if declared_schema is None:
                return "refresh_rejected"
            tool_name = self._schema_tool_name(declared_schema)
            if not tool_name:
                return "refresh_rejected"
            registry_tool = self._full_tool_registry.get(tool_name)
            if registry_tool is None:
                return "refresh_rejected"
            registry_schema = self._schema_dict_from_tool(registry_tool)
            if self._canonical_json(registry_schema) != self._canonical_json(declared_schema):
                return "refresh_rejected"
            if tool_name in candidate_active:
                active_schema = self._schema_dict_from_tool(candidate_active[tool_name])
                if self._canonical_json(active_schema) != self._canonical_json(registry_schema):
                    return "refresh_rejected"
            validated_names.append(tool_name)
            if tool_name not in candidate_active:
                candidate_active[tool_name] = registry_tool
                candidate_order.append(tool_name)

        candidate_schemas = [self._schema_dict_from_tool(candidate_active[name]) for name in candidate_order]
        agent_data._active_tools = candidate_active
        agent_data._active_tool_schemas = copy.deepcopy(candidate_schemas)
        return "refresh_applied" if validated_names else "refresh_no_change"

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _classify_result_status(result_text: str) -> str:
        payload = BusinessToolAdapter._parse_json_object(result_text)
        if isinstance(payload, dict) and payload.get("error"):
            return "tool_error"
        # A returned string proves backend completion, not business API success.
        return "backend_returned"

    @staticmethod
    def _normalize_schema(schema_like: Any) -> dict[str, Any] | None:
        if isinstance(schema_like, OpenAIFunctionToolSchema):
            return schema_like.model_dump(exclude_unset=True, exclude_none=True)
        if isinstance(schema_like, Mapping):
            try:
                # The original HTTPToolManager returns bare function definitions.
                # BaseTool uses the OpenAI {type, function} envelope.
                if "function" not in schema_like and isinstance(schema_like.get("name"), str):
                    schema_like = {"type": "function", "function": dict(schema_like)}
                return OpenAIFunctionToolSchema.model_validate(schema_like).model_dump(
                    exclude_unset=True, exclude_none=True
                )
            except Exception:
                return None
        return None

    @staticmethod
    def _schema_tool_name(schema_dict: Mapping[str, Any]) -> str | None:
        function_block = schema_dict.get("function")
        if not isinstance(function_block, Mapping):
            return None
        name = function_block.get("name")
        return name if isinstance(name, str) and name else None

    @staticmethod
    def _schema_dict_from_tool(tool: Any) -> dict[str, Any]:
        schema = getattr(tool, "tool_schema", None)
        if schema is None and hasattr(tool, "get_openai_tool_schema"):
            schema = tool.get_openai_tool_schema()
        if isinstance(schema, OpenAIFunctionToolSchema):
            return schema.model_dump(exclude_unset=True, exclude_none=True)
        if isinstance(schema, Mapping):
            return OpenAIFunctionToolSchema.model_validate(schema).model_dump(exclude_unset=True, exclude_none=True)
        raise TypeError(f"Tool {tool!r} does not expose a valid OpenAI tool schema.")

    @staticmethod
    def _canonical_json(value: Mapping[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _build_stats(self, *, status: str, text: str) -> StatsDict:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] if text else None
        return {
            "tool_name": self.name,
            "status": status,
            "result_bytes": len(text.encode("utf-8")),
            "result_digest": digest,
        }

    def _record_stats(self, agent_data: Any, stats: StatsDict) -> None:
        if agent_data is None:
            return
        extra_fields = getattr(agent_data, "extra_fields", None)
        if not isinstance(extra_fields, dict):
            return
        rows = extra_fields.setdefault(self._STATS_FIELD, [])
        if isinstance(rows, list):
            rows.append(copy.deepcopy(stats))
