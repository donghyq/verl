"""Install the verified formatter adapter on a newly created business loop.

The native ToolAgentLoop already parses data.response_ids asynchronously and
passes the validated prefix to _build_assistant_message. Keep that parser and
the original reward formatter; do not parse tokens again or edit final answers.
"""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import MethodType

HELPER_SHA256 = "024ed49fd2407df4e304796ad030e5483ab762b48d0437c097b7ea0b44329c1e"


def install_business_protocol_loop(loop, *, parser_type=None):
    if getattr(loop, "business_structured_tool_message_overlay", None) is not None:
        raise ValueError("Structured tool message overlay was already installed")
    if parser_type is None:
        from business_function_parser import BusinessFunctionToolParser
        parser_type = BusinessFunctionToolParser
    if not isinstance(loop.tool_parser, parser_type):
        raise ValueError("The original validated business function parser is required")
    if parser_type.__name__ != "BusinessFunctionToolParser":
        raise ValueError("Unexpected parser class for the original business protocol")
    helper_path = Path(__file__).with_name("structured_tool_message_adapter.py")
    if hashlib.sha256(helper_path.read_bytes()).hexdigest() != HELPER_SHA256:
        raise ValueError("The CPU verified formatter helper changed")
    spec = importlib.util.spec_from_file_location("_formal_reward_r9_tool_message_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original_builder = loop._build_assistant_message
    parser_path = Path(inspect.getfile(parser_type)).resolve()
    receipt = {
        "status": "INSTALLED_ON_NEW_BUSINESS_PROTOCOL_LOOP",
        "overlay_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": HELPER_SHA256,
        "original_parser_source": str(parser_path),
        "original_parser_sha256": hashlib.sha256(parser_path.read_bytes()).hexdigest(),
        "native_final_text_branch_unchanged": True,
        "model_tokens_and_logprobs_unchanged": True,
    }

    def build(self, validated_prefix, data):
        if not data.tool_calls:
            return original_builder(validated_prefix, data)
        schemas = getattr(data, "_active_tool_schemas", self.tool_schemas)
        _, _, reasoning, _ = self.business_manager.parse_response(
            validated_prefix, current_functions=[s.get("function", s) for s in schemas]
        )
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("Original reasoning parser returned a non-text field")
        calls = data.tool_calls[:self.max_parallel_calls]
        parsed = [{"name": c.name, "args": json.loads(c.arguments)} for c in calls]
        message = helper.build_tool_message(
            self.business_reward_runtime.build_assistant_message,
            parsed_tool_calls=parsed,
            validated_prefix=validated_prefix,
            reasoning_content=reasoning,
        )
        for call, declared in zip(calls, message.get("tool_calls", [])):
            call.tool_call_id = declared["id"]
        data.extra_fields.setdefault("business_structured_tool_message_receipts", []).append({
            "helper_sha256": HELPER_SHA256,
            "overlay_sha256": receipt["overlay_sha256"],
            "validated_prefix_sha256": hashlib.sha256(validated_prefix.encode()).hexdigest(),
            "actual_response_ids_sha256": hashlib.sha256(
                json.dumps(data.response_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "structured_tool_calls": len(calls),
            "tool_function_xml_in_content": "<function=" in (message.get("content") or ""),
            "model_tokens_and_logprobs_unchanged": True,
        })
        return message

    loop._build_assistant_message = MethodType(build, loop)
    loop.business_structured_tool_message_overlay = receipt
    return receipt
