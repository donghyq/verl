"""Candidate business wiring for frozen V1; no model, HTTP or job runs at import."""

import copy
from contextvars import ContextVar
import ast
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace


AGENT_NAME = "qwen35_business_trajectory"
BUSINESS_TOOL_DISCLOSURE_PROTOCOL = "append_only_skill_json_fixed_initial_tools_v1"


def load_contract(path):
    contract = json.loads(Path(path).read_text())
    if contract["status"] != "PREPARED_BUSINESS_WIRING_CANDIDATE":
        raise ValueError("Unknown business contract status")
    for row in contract["source_files"]:
        if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Pinned business source differs: " + row["role"])
    return contract


def _support(contract, name):
    root = Path(contract["paths"]["business_support_root"]).resolve()
    expected = root / (name + ".py")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != expected:
        raise ValueError("Business helper imported from another source: " + name)
    return module


def validate_business_record(row, context_extractor):
    """Validate an actual V1 row; do not invent identities or discard messages."""
    for key in ("prompt", "data_source", "agent_name", "reward_model", "extra_info"):
        if key not in row:
            raise ValueError("Actual business row is missing " + key)
    if row["agent_name"] != AGENT_NAME:
        raise ValueError("Business row does not select the declared adapter")
    if not isinstance(row["data_source"], str) or not row["data_source"]:
        raise ValueError("Actual data_source is missing")
    if not isinstance(row["reward_model"], dict) or "ground_truth" not in row["reward_model"]:
        raise ValueError("Preserve the actual reward_model/ground_truth field")
    if not isinstance(row["extra_info"], dict):
        raise ValueError("Actual extra_info must be an object")
    messages = row["prompt"]
    if (not isinstance(messages, list) or not messages or not isinstance(messages[0], dict) or messages[0].get("role") != "system"
            or not isinstance(messages[0].get("content"), str) or not messages[0]["content"]):
        raise ValueError("The actual effective system prompt is required; do not inject it twice")
    if not all(isinstance(m, dict) and m.get("role") in {"system", "user", "assistant", "tool"} for m in messages):
        raise ValueError("Business messages have unsupported roles or shapes")
    context = context_extractor(row.get("business_query_context", row["extra_info"].get("business_query_context", row)))
    if not any(m.get("role") == "user" and m.get("content") == context["query"] for m in messages):
        raise ValueError("Same-record business query does not occur in the actual messages")
    return copy.deepcopy(messages), context


def _record_messages(data, *, complete=False):
    data.extra_fields["messages"] = {"messages": copy.deepcopy(data.messages)}
    data.extra_fields["business_reward_messages_complete"] = complete


def _business_template_tools(messages, tools):
    """Forward selected schemas that the original system has not already declared."""
    if not tools:
        return None
    decoder = json.JSONDecoder()
    embedded = {}
    for message in messages:
        if message.get("role") != "system" or not isinstance(message.get("content"), str):
            continue
        text = message["content"]
        for index, value in enumerate(text):
            if value != "{":
                continue
            try:
                declaration, _ = decoder.raw_decode(text[index:])
            except ValueError:
                continue
            if (isinstance(declaration, dict) and isinstance(declaration.get("name"), str)
                    and isinstance(declaration.get("parameters"), dict)):
                name = declaration["name"]
                if name in embedded and embedded[name] != declaration:
                    raise ValueError("Conflicting original system tool declarations: " + name)
                embedded[name] = declaration
    missing = []
    for schema in tools:
        function = schema.get("function", schema)
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Selected business tool has no function name")
        if name not in embedded:
            missing.append(schema)
    return missing or None


def make_business_loop_class(tool_loop, durable_loop):
    # This middle class is essential: frozen Durable.run calls super()
    # handlers, so ordinary subclass overrides would be bypassed.
    class BusinessProtocolLoop(tool_loop):
        async def ct_build_initial_tokens(self, messages, **kwargs):
            kwargs["tools"] = _business_template_tools(
                messages, json.loads(self.business_template_tool_schemas_json))
            tokens = await super().ct_build_initial_tokens(messages, **kwargs)
            # Observe the actual rendered prefix; never fabricate reasoning text.
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
            self.business_manager.enable_thinking = decoded.rstrip().endswith("<think>")
            return tokens

        async def ct_merge_context_msg(self, previous_messages, *args, **kwargs):
            kwargs["tools"] = _business_template_tools(
                previous_messages, json.loads(self.business_template_tool_schemas_json))
            result = await super().ct_merge_context_msg(previous_messages, *args, **kwargs)
            decoded = self.tokenizer.decode(result[0].token_ids, skip_special_tokens=False)
            self.business_manager.enable_thinking = decoded.rstrip().endswith("<think>")
            return result

        async def _handle_pending_state(self, data, params):
            data.extra_fields["prompts"] = copy.deepcopy(data.messages)
            state = await super()._handle_pending_state(data, params)
            _record_messages(data)
            return state

        def _build_assistant_message(self, content, data):
            schemas = getattr(data, "_active_tool_schemas", self.tool_schemas)
            _, _, reasoning, normal = self.business_manager.parse_response(
                content, current_functions=[s.get("function", s) for s in schemas])
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError("Original reasoning parser returned a non-text field")
            calls = data.tool_calls[:self.max_parallel_calls]
            parsed = [{"name": c.name, "args": json.loads(c.arguments)} for c in calls]
            message = self.business_reward_runtime.build_assistant_message(SimpleNamespace(
                parsed_tool_calls=parsed, normal_text=normal, reasoning_content=reasoning))
            for call, declared in zip(calls, message.get("tool_calls", [])):
                call.tool_call_id = declared["id"]
            return message

        async def _handle_generating_state(self, data, params, ignore_termination=False):
            before = len(data.messages)
            try:
                state = await super()._handle_generating_state(data, params, ignore_termination=ignore_termination)
            except self.business_protocol_error as error:
                # Upstream already retained generated tokens/mask/logprobs. Do
                # not add an unobserved final message or turn this into success.
                data.extra_fields["business_function_format_error"] = str(error)
                data.extra_fields["original_source_terminal_condition"] = "RolloutFormatError"
                from verl.experimental.agent_loop.tool_agent_loop import AgentState
                state = AgentState.TERMINATED
            appended = len(data.messages) > before
            _record_messages(data, complete=appended and getattr(state, "name", "") == "TERMINATED" and not data.tool_calls)
            data.extra_fields["business_budget_exit_before_message_append"] = not appended
            return state

        async def _call_tool(self, tool_call, tools_kwargs, agent_data):
            if tool_call.name != "load_skill":
                return await super()._call_tool(tool_call, tools_kwargs, agent_data)
            # Generic text truncation must not cut the skill body/new_tools JSON.
            # The adapter captures its actual result; no schema/text is synthesized.
            agent_data._business_skill_result_text = None
            try:
                response, reward, stats = await super()._call_tool(tool_call, tools_kwargs, agent_data)
                original = agent_data._business_skill_result_text
                if isinstance(original, str):
                    response.text = original
                return response, reward, stats
            finally:
                del agent_data._business_skill_result_text

        async def _handle_processing_tools_state(self, data):
            # Pinned RLFactoryAgentLoop terminates on the first response or
            # agent_handoff call without executing it. The native ToolAgentLoop
            # has already appended its actual structured assistant message.
            called_tool_name = data.tool_calls[0].name if data.tool_calls else None
            if called_tool_name in ("response", "agent_handoff"):
                from verl.experimental.agent_loop.tool_agent_loop import AgentState
                data.extra_fields["original_source_terminal_condition"] = "terminal_tool:" + called_tool_name
                _record_messages(data, complete=True)
                return AgentState.TERMINATED
            state = await super()._handle_processing_tools_state(data)
            if "trajectory_tool_schema_fingerprint" in data.extra_fields:
                # load_skill changes the actual trajectory-local schemas. Bind
                # the next snapshot to these real schemas, as native restore does.
                from verl.trainer.ppo.v1.trajectory_state.preemptible_tool_agent_loop import _canonical_fingerprint
                data.extra_fields["trajectory_tool_schema_fingerprint"] = _canonical_fingerprint(
                    {"tools": data._active_tool_schemas})
            _record_messages(data)
            return state

    class BusinessTrajectoryLoop(durable_loop, BusinessProtocolLoop):
        def _ct_identity(self):
            identity = super()._ct_identity()
            # Runtime tools grow separately; rendered history keeps one fixed
            # initial header. Native exact identity matching rejects old snapshots.
            identity["business_tool_disclosure_protocol"] = BUSINESS_TOOL_DISCLOSURE_PROTOCOL
            identity["business_template_tools_sha256"] = hashlib.sha256(
                self.business_template_tool_schemas_json.encode("utf-8")).hexdigest()
            return identity

        async def run(self, sampling_params, priority=0, **kwargs):
            sampling_params = dict(sampling_params)
            sampling_params["max_new_tokens"] = self.business_max_new_tokens
            messages = kwargs["raw_prompt"]
            if not messages or not isinstance(messages[0], dict) or messages[0].get("role") != "system":
                raise ValueError("Actual effective business system prompt is required")
            context_source = kwargs.get("business_query_context", (kwargs.get("extra_info") or {}).get("business_query_context", kwargs))
            context = self.business_context_extractor(context_source)
            original_defaults = (kwargs.get("extra_info") or {}).get("default_tool_kwargs", {})
            if not isinstance(original_defaults, dict):
                raise ValueError("Original per-record default_tool_kwargs must be an object")
            tools_kwargs = {name: {"create_kwargs": {"messages": copy.deepcopy(messages),
                "business_query_context": copy.deepcopy(context),
                "original_default_tool_kwargs": copy.deepcopy(original_defaults)}} for name in self.tools}
            if kwargs.get("tools_kwargs") and kwargs["tools_kwargs"] != tools_kwargs:
                raise ValueError("Configured tool context differs from the actual same-record input")
            kwargs["tools_kwargs"] = tools_kwargs
            extra = copy.deepcopy(kwargs.get("extra_info") or {})
            resume = kwargs.get("_trajectory_resume")
            if resume is not None:
                decoded = self.tokenizer.decode(resume.agent_state["prompt_ids"], skip_special_tokens=False)
                self.business_manager.enable_thinking = decoded.rstrip().endswith("<think>")
            selection = list(resume.tool_selection) if resume is not None else self.business_initial_tools
            requested = extra.get("tool_selection")
            if requested is not None and requested not in (self.business_initial_tools, selection):
                raise ValueError("Business input tool selection differs from original/saved state")
            extra["tool_selection"] = selection
            kwargs["extra_info"] = extra
            return await super().run(sampling_params, priority=priority, **kwargs)

    return BusinessTrajectoryLoop


def configure_text_only_qwen_builder(loop):
    """Use Qwen's text protocol for this text-only business contract.

    Qwen3.5 checkpoints also load a vision processor. That capability must not
    select a processor-backed builder for trajectories whose durable state is
    deliberately text-only. Media remain rejected by the durable state guard.
    """
    from omegaconf import OmegaConf
    from verl.utils.tokenizer.continuous_token import (
        QwenContinuousTokenBuilder, QwenVLContinuousTokenBuilder,
    )

    original = loop.continuous_token_builder
    if type(original) not in {QwenContinuousTokenBuilder, QwenVLContinuousTokenBuilder}:
        raise ValueError("Business text contract requires the upstream Qwen token protocol")
    settings = original.chat_template_kwargs
    if OmegaConf.is_config(settings):
        settings = OmegaConf.to_container(settings, resolve=True, throw_on_missing=True)
    loop.continuous_token_builder = QwenContinuousTokenBuilder(
        loop.tokenizer, chat_template_kwargs=copy.deepcopy(settings),
        allowed_append_roles=original.allowed_append_roles,
    )


def create_business_loop(*, business_contract_path, business_auth_file, **kwargs):
    contract = load_contract(business_contract_path)
    paths = contract["paths"]
    parser = _support(contract, "business_function_parser")
    loader = _support(contract, "business_runtime_loader")
    budgets = _support(contract, "business_rollout_contract").parse_rollout_contract(contract["rollout_contract"])
    if contract.get("http_route_mode") != "official_new_route":
        raise ValueError("Current official route requires a frozen actual tool catalog before model use")
    routes = importlib.import_module("business_http_routes")
    if Path(routes.__file__).resolve() != Path(__file__).resolve().parent / "business_http_routes.py":
        raise ValueError("Business route adapter imported from another source")
    business = routes.load_routed_runtime(loader,
        catalog_path=paths["current_tool_catalog"], catalog_receipt_path=paths["current_tool_catalog_receipt"],
        source_root=paths["business_source_root"], http_config_path=paths["http_config"],
        agent_config_path=paths["agent_config"], auth_file=business_auth_file, rollout_contract=budgets)
    business.manager.tool_call_parser = "qwen3_coder"
    business.manager.reasoning_parser = "qwen3"
    reward = _support(contract, "business_reward_runtime").load_original_reward_runtime(paths["original_reward_root"])
    adapter = _support(contract, "business_tool_adapter").BusinessToolAdapter
    schema_view = _support(contract, "business_schema_view").OriginalBusinessSchemaView
    from verl.experimental.agent_loop.agent_loop import ToolListWrap
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
    from verl.trainer.ppo.v1.trajectory_state.preemptible_tool_agent_loop import DurableToolAgentLoop
    class BusinessSchemaToolAdapter(adapter):
        async def execute(self, instance_id, parameters, **kwargs):
            data = kwargs["agent_data"]
            # tools_kwargs is an existing durable snapshot field. Read the
            # restored same-record value, never reconstruct it from the query.
            defaults = data.tools_kwargs[self.name]["create_kwargs"]["original_default_tool_kwargs"]
            if not isinstance(defaults, dict):
                raise ValueError("Saved original tool defaults must be an object")
            token = tool_defaults.set(copy.deepcopy(defaults))
            try:
                result = await super().execute(instance_id, parameters, **kwargs)
                if self.name == "load_skill":
                    data._business_skill_result_text = result[0].text
                return result
            finally:
                tool_defaults.reset(token)

        # The factory deliberately uses a lossless view. Keep that exact view
        # through trajectory-local load_skill validation; canonical comparison
        # with the pinned registry below still rejects changed schemas.
        @staticmethod
        def _schema_dict_from_tool(tool):
            schema = getattr(tool, "tool_schema", None)
            if isinstance(schema, schema_view):
                return schema.model_dump(exclude_unset=True, exclude_none=True)
            return adapter._schema_dict_from_tool(tool)

        @staticmethod
        def _normalize_schema(schema):
            if isinstance(schema, (dict, schema_view)):
                try:
                    view = schema if isinstance(schema, schema_view) else schema_view(schema)
                    return view.model_dump(exclude_unset=True, exclude_none=True)
                except (TypeError, ValueError):
                    return None
            return adapter._normalize_schema(schema)

    tool_defaults = ContextVar("business_original_default_tool_kwargs", default=None)

    def call_original_manager(tool_name, parameters, **_bridge_kwargs):
        defaults = tool_defaults.get()
        if defaults is None:
            raise ValueError("Original per-record tool defaults are not bound to this trajectory")
        # The pinned manager applies its own ignored-key filter and merge order.
        return business.manager.call_tool(tool_name, parameters, tool_kwargs=copy.deepcopy(defaults))

    registry = {name: BusinessSchemaToolAdapter({}, schema_view(schema), manager_call_tool=call_original_manager,
        full_tool_registry={}, backend_parameter_schema=business.api_parameter_schemas.get(name))
        for name, schema in business.function_schemas.items()}
    for tool in registry.values():
        tool._full_tool_registry = dict(registry)
    kwargs["tools"] = ToolListWrap(list(registry.values()))
    cls = make_business_loop_class(ToolAgentLoop, DurableToolAgentLoop)
    loop = cls(**kwargs)
    configure_text_only_qwen_builder(loop)
    loop.business_manager = business.manager
    loop.business_reward_runtime = reward
    loop.business_context_extractor = _support(contract, "business_query_context").extract_business_query_context
    loop.business_protocol_error = parser.BusinessFunctionProtocolError
    # Pinned RLFactory.init_class uses manager.functions, not agent.tools.disclosed.
    loop.business_initial_tools = [schema.get("function", schema)["name"] for schema in business.manager.functions]
    if (len(set(loop.business_initial_tools)) != len(loop.business_initial_tools)
            or set(loop.business_initial_tools) - set(registry)):
        raise ValueError("Original manager tool set is not a unique subset of the pinned registry")

    if loop.business_initial_tools != business.initial_tool_names:
        raise ValueError("Business initial-tool receipt differs from actual original manager.functions")
    # Keep template tools immutable. Full skill content/new_tools already reach
    # the append-only tool message; runtime parsing/dispatch still use _active_tools.
    loop.business_template_tool_schemas_json = json.dumps(
        [registry[name].tool_schema.model_dump(exclude_unset=True, exclude_none=True)
         for name in loop.business_initial_tools],
        sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    loop.business_max_new_tokens = budgets.max_new_tokens
    _support(contract, 'business_structured_tool_message_overlay').install_business_protocol_loop(
        loop, parser_type=parser.BusinessFunctionToolParser)
    return loop


class _AttributeConfig(dict):
    def __getattr__(self, name):
        return self[name]


async def call_original_score(runtime, config, *, data_source, solution_str, ground_truth, extra_info,
                              reward_router_address=None, reward_model_tokenizer=None, tokenizer=None):
    extra = copy.deepcopy(extra_info)
    if not isinstance(extra.get("messages"), dict) or set(extra["messages"]) != {"messages"}:
        raise ValueError("The original messages wrapper is missing")
    if "prompts" not in extra:
        raise ValueError("Actual prompt messages are missing")
    if "reward_config" in extra and any(extra["reward_config"].get(k) != v for k, v in config.items()):
        raise ValueError("Input cannot silently replace the pinned original reward configuration")
    extra["reward_config"] = _AttributeConfig(config)
    # The original DAPO manager supplies default_local_dir, while the pinned
    # scorer reads only reward_log_base_dir. Bridge these logging fields here.
    # Explicit env/config destinations and disabled logging retain priority.
    log_env = dict(extra.get("env") or {})
    log_enabled = log_env.get("reward_log_enabled", config.get("reward_log_enabled", True))
    if str(log_enabled).lower() in ("true", "1", "yes", "on"):
        log_dir = log_env.get("reward_log_base_dir") or config.get("reward_log_base_dir")
        if not log_dir:
            native_dir = extra.get("default_local_dir")
            if not native_dir:
                raise ValueError("Enabled original reward logging requires default_local_dir or reward_log_base_dir")
            log_dir = str(Path(native_dir) / "reward_logs")
        log_env["reward_log_base_dir"] = str(Path(log_dir))
        extra["env"] = log_env
    result = await runtime.compute_score_server(data_source, solution_str, ground_truth, extra,
        reward_router_address, reward_model_tokenizer, tokenizer)
    if (not isinstance(result, dict) or isinstance(result.get("score"), bool)
            or not isinstance(result.get("score"), (int, float)) or not math.isfinite(result["score"])):
        raise ValueError("Original reward result has no finite numeric score")
    return result  # Retain -1, original errors and all score components unchanged.


async def compute_score(data_source, solution_str, ground_truth, extra_info, *, business_contract_path, **kwargs):
    """Match native V1 NaiveRewardManager kwargs; delegate the pinned original scorer."""
    contract = load_contract(business_contract_path)
    runtime = _support(contract, "business_reward_runtime").load_original_reward_runtime(contract["paths"]["original_reward_root"])
    config = json.loads(Path(contract["paths"]["reward_config"]).read_text())
    return await call_original_score(runtime, config, data_source=data_source, solution_str=solution_str,
        ground_truth=ground_truth, extra_info=extra_info, reward_router_address=kwargs.get("reward_router_address"),
        reward_model_tokenizer=kwargs.get("reward_model_tokenizer"), tokenizer=kwargs.get("tokenizer"))


def make_business_reward_manager(base_class, default_score, source, filename):
    """Reuse the original class, including decode/overlong logic, without registration."""
    tree = ast.parse(source, filename=filename)
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DAPOJudgeRewardLoopManager"]
    if len(nodes) != 1:
        raise ValueError("Pinned original DAPO manager class is missing")
    node = nodes[0]
    node.decorator_list = []  # Do not register private code in the community registry.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"RewardManagerBase": base_class, "default_compute_score": default_score,
                 "inspect": inspect, "__name__": __name__}
    exec(compile(module, filename, "exec"), namespace)
    original = namespace["DAPOJudgeRewardLoopManager"]

    class BusinessRewardManager(original):
        async def run_single(self, data):
            # V1 stores loop messages in tool_extra_fields. The original manager
            # expects messages at the row root. Copy only this interface; leave
            # original token decoding, server scoring and overlong penalty intact.
            converted = copy.copy(data)
            converted.non_tensor_batch = copy.deepcopy(data.non_tensor_batch)
            fields = converted.non_tensor_batch.get("tool_extra_fields")
            if fields is not None:
                messages = [row["messages"] for row in fields]
                old = converted.non_tensor_batch.get("messages")
                if old is not None and list(old) != messages:
                    raise ValueError("Two different original message sources cannot be scored")
                import numpy as np
                converted.non_tensor_batch["messages"] = np.asarray(messages, dtype=object)
            if "messages" not in converted.non_tensor_batch:
                raise ValueError("Actual original structured messages are required")
            return await super().run_single(converted)

    return BusinessRewardManager


def __getattr__(name):
    if name != "BusinessRewardManager":
        raise AttributeError(name)
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "payload-source-manifest.json").read_text())
    row = next(r for r in manifest["files"] if r["role"] == "original_reward_manager")
    path = root / row["relative_path"]
    source = path.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != row["sha256"]:
        raise ValueError("Original DAPO manager bytes differ")
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
    from verl.utils.reward_score import default_compute_score
    cls = make_business_reward_manager(RewardManagerBase, default_compute_score, source, str(path))
    globals()[name] = cls
    return cls
