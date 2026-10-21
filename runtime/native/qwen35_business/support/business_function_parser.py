"""Accept the original business template's bare function XML without changing tokens."""
import asyncio
import json
import re

from verl.experimental.agent_loop.tool_parser import Qwen3XMLToolParser, ToolParser


class BusinessFunctionProtocolError(ValueError):
    """A generated function call violates the declared business format."""


@ToolParser.register("business_function_xml")
class BusinessFunctionToolParser(Qwen3XMLToolParser):
    async def extract_tool_calls(self, responses_ids, tools=None):
        text = await asyncio.get_running_loop().run_in_executor(None, self.tokenizer.decode, responses_ids)
        if "<function=" not in text:
            if "<tool_call>" in text:
                raise BusinessFunctionProtocolError("Incomplete business tool call.")
            return text, []
        matches = list(re.finditer(r"<function=([^>]+)>(.*?)</function>", text, re.DOTALL))
        if len(matches) != text.count("<function="):
            raise BusinessFunctionProtocolError("Incomplete business function call.")
        allowed = {schema.function.name: schema for schema in tools or []}
        calls = []
        for match in matches:
            name, body = match.group(1), match.group(2)
            if name not in allowed:
                raise BusinessFunctionProtocolError("Business function is not in this trajectory's active tool set.")
            parameter_matches = list(re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.DOTALL))
            if len(parameter_matches) != body.count("<parameter="):
                raise BusinessFunctionProtocolError("Incomplete business function parameter.")
            names = [parameter.group(1) for parameter in parameter_matches]
            if len(names) != len(set(names)):
                raise BusinessFunctionProtocolError("Duplicate business function parameter.")
            schema = allowed[name].function.parameters
            if set(names) - set(schema.properties):
                raise BusinessFunctionProtocolError("Business function parameter is not declared.")
            # Missing required arguments are a recoverable model error. The
            # adapter checks the actual backend schema before any HTTP call
            # and returns precise feedback; never fill a missing value here.
            # Reuse the existing typed XML conversion; do not fabricate arguments.
            call = self._parse_xml_function_call(name + ">" + body, tools)
            if call is None or set(json.loads(call.arguments)) != set(names):
                raise BusinessFunctionProtocolError("Business function parameter parsing did not complete.")
            calls.append(call)
        prefix = text.split("<tool_call>", 1)[0] if "<tool_call>" in text else text[:matches[0].start()]
        return prefix, calls
