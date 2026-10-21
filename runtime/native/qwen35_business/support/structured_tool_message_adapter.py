"""Use the validated parser prefix when building a structured tool message.

The original parser has already validated tool names and argument conversion.
Only the message representation changes; model tokens and probabilities stay
unchanged. Pass the prefix returned by BusinessFunctionToolParser, never the
complete decoded function XML. This does not repair generated final answers.
"""
from types import SimpleNamespace


def build_tool_message(original_builder, *, parsed_tool_calls, validated_prefix,
                       reasoning_content):
    if not parsed_tool_calls or "<function=" in validated_prefix or "<tool_call>" in validated_prefix:
        raise ValueError("A validated prefix without function XML is required")
    if validated_prefix.count("</think>") > 1:
        raise ValueError("Ambiguous reasoning boundary in validated prefix")
    normal_text = validated_prefix.split("</think>", 1)[-1]
    if "<think>" in normal_text or "</think>" in normal_text:
        raise ValueError("Incomplete reasoning boundary in validated prefix")
    return original_builder(SimpleNamespace(
        parsed_tool_calls=parsed_tool_calls,
        normal_text=normal_text.strip(),
        reasoning_content=reasoning_content,
    ))
