# Copyright 2026 Project B contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text-only safe-point hooks around the current upstream ToolAgentLoop.

The ordinary run delegates to the upstream implementation. The opt-in path
retains its pending/generation/tool handlers and Continuous Token builder; only
ownership, actual generation provenance, and committed tool-boundary state are
added. A snapshot is never created while a tool batch is executing.
"""

from __future__ import annotations

import inspect
import math
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.tokenizer import continuous_token as ct_module

from .identity import _canonical_fingerprint, _decode_policy_identity
from .policy_publication import PublishedPolicyRecord
from .snapshot import TrajectorySnapshot, canonical_bytes, _is_text_only_message

_TEXT_BUILDERS = frozenset({
    "ContinuousTokenBuilder", "GptOssContinuousTokenBuilder", "QwenContinuousTokenBuilder",
    "MiniMaxContinuousTokenBuilder", "GLMContinuousTokenBuilder", "Gemma4ContinuousTokenBuilder",
    "DeepSeekContinuousTokenBuilder",
})
_SNAPSHOT_FIELDS = (
    "messages", "prompt_ids", "response_ids", "response_mask", "response_logprobs",
    "turn_scores", "tool_rewards", "user_turns", "assistant_turns", "extra_fields",
    "tools_kwargs", "metrics", "request_id", "mm_processor_kwargs",
)


class TrajectoryCheckpointYield(Exception):
    """A committed execution snapshot, never an AgentLoopOutput or trainable sample."""

    def __init__(self, snapshot: TrajectorySnapshot, reference: str):
        super().__init__("trajectory yielded after committing an execution snapshot")
        self.snapshot = snapshot
        self.reference = reference


class TrajectoryPendingYield(Exception):
    """A waiting first generation with no model/tool work; never a sample."""

    def __init__(self, trajectory_id):
        self.trajectory_id = trajectory_id


@dataclass
class _RunContext:
    loop: "DurableToolAgentLoop"
    identity: dict[str, str]
    execution_generation: int
    output: Any = None
    segments: list[dict] | None = None
    policy_resolver: Any = None
    source_backend: Any = None
    source_lease: Any = None
    checkpoint: Any = None
    check_owner: Any = None
    before_generate: Any = None


_RUN_CONTEXT: ContextVar[_RunContext | None] = ContextVar("durable_tool_agent_run", default=None)


class _ObservedServerManager:
    """Observe the real result without replacing a shared manager's methods."""

    def __init__(self, manager, loop):
        self._manager, self._loop = manager, loop

    def __getattr__(self, name):
        return getattr(self._manager, name)

    async def generate(self, *args, **kwargs):
        context = _RUN_CONTEXT.get()
        if context is not None and context.loop is self._loop and context.source_backend is not None:
            if args or context.before_generate is None:
                raise ValueError("token-KV source requires a real pre-generate AgentData capture")
            output = await context.source_backend.generate_owned(
                kwargs, before_generate=context.before_generate, lease=context.source_lease,
                commit_snapshot=context.checkpoint, check_owner=context.check_owner,
            )
        else:
            output = await self._manager.generate(*args, **kwargs)
        if context is not None and context.loop is self._loop:
            published = None
            if context.policy_resolver is not None:
                attempts = output.extra_fields.get("native_generation_attempts", output.extra_fields.get("native_gateway_attempts"))
                if attempts is None and "native_meta_info" in output.extra_fields:
                    attempts = [output.extra_fields]
                if not isinstance(attempts, list) or not attempts:
                    raise ValueError("continuous publication requires actual native generation receipts")
                versions = {_weight_version(row["native_meta_info"]["weight_version"]) for row in attempts}
                published = {version: PublishedPolicyRecord.decode(await context.policy_resolver(version))
                             for version in versions}
            context.segments = _generation_segments(output, kwargs["sampling_params"], context.identity, published)
            context.output = output
        return output


def _weight_version(value: Any) -> str:
    if type(value) is int and value >= 0:
        return str(value)
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return str(int(value))
    raise ValueError("generation requires an observed non-negative policy version")


def _generation_segments(output, sampling_params, identity, published_policies=None) -> list[dict]:
    """Use actual gateway spans, or a covering single-version server receipt.

    The default fixed-window adapter accepts one policy per generate call.
    Continuous publication is opt-in: each native attempt still has one actual
    policy, while different attempts require their own immutable published
    records. No attempt's final version is applied to earlier generated tokens.
    """
    tokens, logprobs = output.token_ids, output.log_probs
    if (logprobs is None or len(logprobs) != len(tokens) or any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in logprobs)):
        raise ValueError("handoff requires actual finite behavior logprobs for every generated token")
    if output.routed_experts is not None:
        raise ValueError("trajectory handoff does not support routed-expert replay")
    canonical_bytes(sampling_params)
    fingerprint = _canonical_fingerprint(sampling_params)
    expected = _weight_version(identity["policy_version"])
    extra = output.extra_fields
    attempts = extra.get("native_generation_attempts", extra.get("native_gateway_attempts"))
    if attempts is None and "native_meta_info" in extra:
        attempts = [{**extra, "output_start": 0, "output_end": len(tokens)}]
    ranges = []
    if attempts is not None:
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("native generation receipts are missing")
        cursor = 0
        for attempt in attempts:
            start, end = attempt.get("output_start"), attempt.get("output_end")
            if (type(start) is not int or type(end) is not int or start != cursor
                    or not start <= end <= len(tokens)):
                raise ValueError("native generation receipt ranges do not cover actual output tokens")
            meta = attempt.get("native_meta_info")
            if not isinstance(meta, dict):
                raise ValueError("native generation receipt has no actual meta_info")
            scalar = _weight_version(meta.get("weight_version"))
            spans = meta.get("weight_versions")
            local_cursor, observed = 0, set()
            if spans is not None:
                if not isinstance(spans, list) or not spans:
                    raise ValueError("native weight-version spans are missing")
                for span in spans:
                    left, right = span.get("start"), span.get("end")
                    if (type(left) is not int or type(right) is not int or left != local_cursor
                            or not left <= right <= end - start or (right == left and end > start)):
                        raise ValueError("native weight-version spans do not cover one attempt")
                    local_cursor = right
                    observed.add(_weight_version(span.get("version")))
                if local_cursor != end - start or observed != {scalar}:
                    raise ValueError("a native generation attempt has inconsistent policy versions")
            if published_policies is not None:
                record = published_policies.get(scalar)
                if (not isinstance(record, PublishedPolicyRecord) or record.identity.policy_version != scalar
                        or record.identity.tokenizer_template_fingerprint != identity["tokenizer_template_fingerprint"]
                        or record.identity.kv_cache_schema_fingerprint != identity["kv_cache_schema_fingerprint"]):
                    raise ValueError("generation has no compatible published policy identity")
                if len({item.run_id for item in published_policies.values()}) != 1:
                    raise ValueError("generation receipts cross policy publication runs")
            elif scalar != expected:
                raise ValueError("generation policy differs from the supplied canonical policy identity")
            if end > start:
                ranges.append((start, end, scalar))
            cursor = end
        if cursor != len(tokens):
            raise ValueError("native generation receipts do not cover the complete output")
    else:
        minimum = _weight_version(extra.get("min_global_steps", extra.get("global_steps")))
        maximum = _weight_version(extra.get("max_global_steps", extra.get("global_steps")))
        if minimum != maximum or minimum != expected:
            raise ValueError("single-policy generation provenance is missing or inconsistent")
        if tokens:
            ranges.append((0, len(tokens), minimum))
    segments = [{
        "start": start, "end": end, "policy_version": version,
        "sampling_config_fingerprint": fingerprint,
    } for start, end, version in ranges]
    if published_policies is not None:
        for segment in segments:
            segment["policy_identity"] = vars(published_policies[segment["policy_version"]].identity).copy()
    return segments


def _json_builder_constant(value):
    if isinstance(value, (set, frozenset)):
        return sorted(_json_builder_constant(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_json_builder_constant(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_builder_constant(item) for key, item in value.items()}
    canonical_bytes(value)  # Reject runtime handles, opaque state and nonfinite values.
    return value


def _generated_history(agent_data):
    mask, logs = agent_data.response_mask, agent_data.response_logprobs
    if len(logs) != len(mask):
        raise ValueError("actual token mask and behavior logprob counts differ")
    response = agent_data.prompt_ids[len(agent_data.prompt_ids) - len(mask):] if mask else []
    return [(token, logprob) for token, bit, logprob in zip(response, mask, logs, strict=True) if bit]


@register("durable_tool_agent")
class DurableToolAgentLoop(ToolAgentLoop):
    """Opt-in text trajectory handoff with unchanged upstream tool execution."""

    def _trajectory_limits(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "response_length", "max_user_turns", "max_assistant_turns", "max_parallel_calls",
            "max_tool_response_length", "tool_response_truncate_side",
        )}

    def _ct_identity(self) -> dict:
        builder = self.continuous_token_builder
        cls = type(builder)
        if (cls.__module__ != ct_module.__name__ or cls.__name__ not in _TEXT_BUILDERS
                or builder.supports_multimodal()):
            raise ValueError("handoff requires an audited upstream text Continuous Token builder")
        constants = {name: _json_builder_constant(value) for name, value in vars(builder).items()
                     if name != "tokenizer"}
        return {
            "builder": f"{cls.__module__}.{cls.__qualname__}",
            "implementation": _canonical_fingerprint({"source": inspect.getsource(ct_module)}),
            "constants": constants,
            "allowed_append_roles": sorted(builder.allowed_append_roles),
            "prompt_length": self.prompt_length,
            "tool_parser": self.tool_parser_name,
            "parser_stop_token_ids": list(self.tool_parser.stop_token_ids or []),
        }

    @staticmethod
    def _check_text_state(agent_data):
        if (agent_data.image_data or agent_data.video_data or agent_data.audio_data
                or agent_data.mm_processor_output is not None or agent_data.routed_experts is not None
                or agent_data.mm_processor_kwargs or any(
                    not _is_text_only_message(message) for message in agent_data.messages)):
            raise ValueError("trajectory snapshots support text-only state without router replay")

    def _select_tools(self, agent_data, selection):
        if selection is None:
            selected = self.tools
        else:
            if (not isinstance(selection, list) or len(set(selection)) != len(selection)
                    or any(name not in self.tools for name in selection)):
                raise ValueError("snapshot tool selection is unknown or duplicated")
            selected = {name: self.tools[name] for name in selection}
        agent_data._active_tools = selected
        agent_data._active_tool_schemas = [
            tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in selected.values()
        ]

    def _make_trajectory_snapshot(self, agent_data, sampling_params, kwargs):
        self._check_text_state(agent_data)
        snapshot = TrajectorySnapshot(
            trajectory_id=kwargs["trajectory_id"], group_id=kwargs["rollout_group_id"],
            member_index=kwargs["rollout_sample_index"], group_size=kwargs["rollout_group_size"],
            execution_generation=kwargs["execution_generation"],
            agent_state={name: getattr(agent_data, name) for name in _SNAPSHOT_FIELDS},
            sampling_params=dict(sampling_params), limits=self._trajectory_limits(),
            environment_version=kwargs["snapshot_environment_version"],
            tool_version=kwargs["snapshot_tool_version"], policy_identity=kwargs["snapshot_policy_identity"],
            tool_selection=list(agent_data._active_tools),
            completed_tool_batches=agent_data.completed_tool_batches,
        )
        return TrajectorySnapshot.decode(snapshot.encode())

    async def run_until_checkpoint(self, sampling_params, *, commit_snapshot, check_owner, **kwargs):
        if not callable(check_owner) or not callable(commit_snapshot):
            raise ValueError("checkpoint execution requires ownership and commit callbacks")
        kwargs.setdefault("_trajectory_policy_resolver", getattr(
            commit_snapshot, "resolve_published_policy", getattr(check_owner, "resolve_published_policy", None)))
        kwargs.setdefault("_trajectory_snapshot_identity_provider",
                          getattr(commit_snapshot, "get_snapshot_policy_identity", None))
        return await self.run(sampling_params, _trajectory_checkpoint=commit_snapshot,
                              _trajectory_check_owner=check_owner, **kwargs)

    async def resume_from_snapshot(self, snapshot, *, check_owner, execution_generation,
                                   environment_version, tool_version, **kwargs):
        if not callable(check_owner):
            raise ValueError("resumed execution requires an ownership callback")
        kwargs.setdefault("snapshot_policy_identity", snapshot.policy_identity)
        kwargs.setdefault("_trajectory_policy_resolver", getattr(check_owner, "resolve_published_policy", None))
        if getattr(check_owner, "token_only_resume", False):
            if kwargs["snapshot_policy_identity"] != snapshot.policy_identity:
                raise ValueError("token-only resume must retain the exact committed source identity")
            target = _decode_policy_identity(check_owner.target_policy_identity)
            if target.tokenizer_template_fingerprint != snapshot.policy_identity["tokenizer_template_fingerprint"]:
                raise ValueError("token-only target tokenizer/template identity differs")
            if kwargs["_trajectory_policy_resolver"] is None:
                raise ValueError("token-only new-policy resume requires a committed publication resolver")
            observed = PublishedPolicyRecord.decode(await kwargs["_trajectory_policy_resolver"](target.policy_version))
            if observed.identity != target:
                raise ValueError("token-only target differs from actual policy publication")
            kwargs["_trajectory_execution_policy_identity"] = vars(target).copy()
            kwargs["_trajectory_token_only_resume"] = True
        return await self.run(snapshot.sampling_params, _trajectory_resume=snapshot,
                              _trajectory_check_owner=check_owner,
                              snapshot_environment_version=environment_version,
                              snapshot_tool_version=tool_version,
                              execution_generation=execution_generation, **kwargs)

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        controls = ("_trajectory_resume", "_trajectory_checkpoint", "_trajectory_check_owner")
        if not any(kwargs.get(key) is not None for key in controls):
            return await super().run(sampling_params, priority=priority, **kwargs)
        check_owner = kwargs.get("_trajectory_check_owner")
        if not callable(check_owner):
            raise ValueError("stateful execution requires an ownership callback")
        identity = dict(kwargs["snapshot_policy_identity"])
        _decode_policy_identity(identity)
        generation = kwargs["execution_generation"]
        if type(generation) is not int or generation < 0:
            raise ValueError("snapshot execution generation must be non-negative")
        canonical_bytes(sampling_params)
        checkpoint = kwargs.get("_trajectory_checkpoint")
        if checkpoint is not None and not callable(checkpoint):
            raise ValueError("snapshot commit callback must be callable")
        acceptance_boundary = kwargs.get("_trajectory_acceptance_tool_boundary")
        if acceptance_boundary is not None and not callable(acceptance_boundary):
            raise ValueError("acceptance tool-boundary callback must be callable")
        batches = kwargs.get("_trajectory_checkpoint_after_batches", 1)
        if type(batches) is not int or batches < 1:
            raise ValueError("checkpoint batch count must be a positive integer")
        ct_identity = self._ct_identity()
        resume = kwargs.get("_trajectory_resume")
        resume = TrajectorySnapshot.decode(resume.encode()) if resume is not None else None
        if resume is None:
            for name in ("trajectory_id", "rollout_group_id", "snapshot_environment_version", "snapshot_tool_version"):
                if not isinstance(kwargs.get(name), str) or not kwargs[name]:
                    raise ValueError(f"stateful trajectory requires {name}")
            index, size = kwargs.get("rollout_sample_index"), kwargs.get("rollout_group_size")
            if type(size) is not int or size < 1 or type(index) is not int or not 0 <= index < size:
                raise ValueError("stateful trajectory membership is invalid")
        messages = deepcopy(resume.agent_state["messages"] if resume else kwargs["raw_prompt"])
        data = AgentData(messages, None, None, None, {}, {},
                         self._make_request_id(priority, self.rollout_config.full_determinism),
                         deepcopy(kwargs.get("tools_kwargs", {})))
        data.completed_tool_batches = []
        self._check_text_state(data)
        if resume is not None:
            if generation <= resume.execution_generation:
                raise ValueError("target execution generation must advance beyond the committed source")
            for name, value in {
                "trajectory_id": resume.trajectory_id, "rollout_group_id": resume.group_id,
                "rollout_sample_index": resume.member_index, "rollout_group_size": resume.group_size,
            }.items():
                if name in kwargs and kwargs[name] != value:
                    raise ValueError("restored trajectory identity/member cannot change")
                kwargs[name] = value
            if resume.limits != self._trajectory_limits() or resume.sampling_params != sampling_params:
                raise ValueError("restored trajectory budgets/sampling configuration changed")
            if (resume.environment_version != kwargs["snapshot_environment_version"]
                    or resume.tool_version != kwargs["snapshot_tool_version"]):
                raise ValueError("restored trajectory tool/environment version mismatch")
            if resume.policy_identity["tokenizer_template_fingerprint"] != identity["tokenizer_template_fingerprint"]:
                raise ValueError("restored trajectory tokenizer/template identity differs")
            if resume.agent_state["extra_fields"].get("trajectory_ct_identity") != ct_identity:
                raise ValueError("restored Continuous Token implementation/config identity differs")
            for name, value in resume.agent_state.items():
                setattr(data, name, value)
            data.completed_tool_batches = list(resume.completed_tool_batches)
            data.tool_calls = []
            requested = (kwargs.get("extra_info") or {}).get("tool_selection")
            if requested is not None and requested != resume.tool_selection:
                raise ValueError("restored tool selection cannot change")
            self._select_tools(data, resume.tool_selection)
            if data.extra_fields.get("trajectory_tool_schema_fingerprint") != _canonical_fingerprint(
                    {"tools": data._active_tool_schemas}):
                raise ValueError("restored tool schema identity differs")
        else:
            self._select_tools(data, (kwargs.get("extra_info") or {}).get("tool_selection"))
            data.extra_fields["trajectory_ct_identity"] = ct_identity
            data.extra_fields["trajectory_tool_schema_fingerprint"] = _canonical_fingerprint(
                {"tools": data._active_tool_schemas})
        self._check_text_state(data)
        # The adapter wraps only this loop's reference; a shared manager is untouched.
        if not isinstance(self.server_manager, _ObservedServerManager):
            self.server_manager = _ObservedServerManager(self.server_manager, self)
        execution_identity = kwargs.get("_trajectory_execution_policy_identity", identity)
        resolver = kwargs.get("_trajectory_policy_resolver")
        if resolver is not None and not callable(resolver):
            raise ValueError("policy publication resolver must be callable")
        context = _RunContext(
            self, execution_identity, generation, policy_resolver=resolver,
            source_backend=kwargs.get("_trajectory_token_kv_source_backend"),
            source_lease=kwargs.get("_trajectory_source_lease"), checkpoint=checkpoint, check_owner=check_owner,
        )
        if kwargs.get("_trajectory_token_only_resume"):
            # Exact history is restored above. The adapter never imports kv_ref
            # or re-renders the saved messages to obtain prompt tokens.
            data.extra_fields["trajectory_resume_mode"] = "token_only"
        handle = _RUN_CONTEXT.set(context)
        try:
            state = AgentState.GENERATING if resume is not None else AgentState.PENDING
            while state != AgentState.TERMINATED:
                await check_owner()
                if state == AgentState.PENDING:
                    state = await super()._handle_pending_state(data, sampling_params)
                elif state == AgentState.GENERATING:
                    if context.source_backend is not None:
                        from .token_kv_resume import capture_before_generate_snapshot

                        context.before_generate = capture_before_generate_snapshot(
                            self, data, sampling_params=sampling_params, run_kwargs=kwargs,
                        )
                    before = _generated_history(data)
                    previous_segments = deepcopy(data.extra_fields.get("trajectory_segments", []))
                    context.output, context.segments = None, None
                    state = await super()._handle_generating_state(data, sampling_params)
                    if context.output is None:
                        raise ValueError("generation completed without an observed token result")
                    if (resume is not None and resume.schema_version == 2
                            and "trajectory_inflight_generation" in data.extra_fields):
                        from .token_kv_resume import resumed_generation_segments

                        prefix = data.extra_fields["trajectory_inflight_generation"]["output_ids"]
                        context.segments = resumed_generation_segments(
                            context.segments, prefix_tokens=len(prefix),
                            source_generation=resume.execution_generation, target_generation=generation,
                        )
                        # The original handler merged the complete saved prefix plus
                        # tail once. Later tool-boundary snapshots carry no unfinished turn.
                        data.extra_fields.pop("trajectory_inflight_generation", None)
                    output = context.output
                    if _generated_history(data) != before + list(zip(output.token_ids, output.log_probs, strict=True)):
                        raise ValueError("Continuous Token merge changed historical generated tokens or logprobs")
                    offset = len(before)
                    for segment in context.segments:
                        previous_segments.append({**segment, "start": offset + segment["start"],
                                                  "end": offset + segment["end"],
                                                  "turn_index": data.assistant_turns - 1,
                                                  "execution_generation": segment.get("execution_generation", generation)})
                    data.extra_fields["trajectory_segments"] = previous_segments
                    for key, value in output.extra_fields.items():
                        if key != "trajectory_segments":
                            data.extra_fields.setdefault(key, deepcopy(value))
                    data.extra_fields.setdefault("trajectory_generation_receipts", []).append({
                        "turn_index": data.assistant_turns - 1, "execution_generation": generation,
                        "generated_start": offset, "generated_end": len(_generated_history(data)),
                        "extra_fields": deepcopy(output.extra_fields),
                    })
                elif state == AgentState.PROCESSING_TOOLS:
                    before = _generated_history(data)
                    message_count = len(data.messages)
                    calls = [{"name": call.name, "arguments": call.arguments}
                             for call in data.tool_calls[:self.max_parallel_calls]]
                    state = await super()._handle_processing_tools_state(data)
                    self._check_text_state(data)
                    if state == AgentState.GENERATING:
                        if _generated_history(data) != before:
                            raise ValueError("tool-context merge changed historical generated tokens or logprobs")
                        results = deepcopy(data.messages[message_count:])
                        if not calls or len(results) != len(calls) or any(row.get("role") != "tool" for row in results):
                            raise ValueError("completed tool batch does not contain all corresponding results")
                        data.completed_tool_batches.append({
                            "turn_index": data.assistant_turns - 1, "calls": calls, "results": results,
                        })
                        await check_owner()
                        if acceptance_boundary is not None:
                            # A functional fixture may wait at this completed tool
                            # batch. It cannot inspect or alter mutable AgentData.
                            await acceptance_boundary(
                                data.assistant_turns - 1, len(data.completed_tool_batches)
                            )
                            await check_owner()
                        if checkpoint is not None and len(data.completed_tool_batches) >= batches:
                            provider = kwargs.get("_trajectory_snapshot_identity_provider")
                            if provider is not None:
                                snapshot_identity = await provider()
                                if snapshot_identity is None:
                                    continue  # No preemption requested; normal upstream execution.
                                kwargs["snapshot_policy_identity"] = snapshot_identity
                            snapshot = self._make_trajectory_snapshot(data, sampling_params, kwargs)
                            reference = await checkpoint(snapshot)
                            if reference is not None:
                                raise TrajectoryCheckpointYield(snapshot, reference)
                else:
                    raise ValueError(f"invalid trajectory execution state: {state}")
            await check_owner()
            return self._finished_output(data, generation, kwargs)
        finally:
            _RUN_CONTEXT.reset(handle)

    def _finished_output(self, data, generation, kwargs):
        self._check_text_state(data)
        size = len(data.response_mask)
        response = data.prompt_ids[len(data.prompt_ids) - size:] if size else []
        prompt = data.prompt_ids[:len(data.prompt_ids) - size] if size else data.prompt_ids
        mask, logs = data.response_mask[:self.response_length], data.response_logprobs[:self.response_length]
        retained = sum(mask)
        data.extra_fields["trajectory_segments"] = [
            {**segment, "end": min(segment["end"], retained)}
            for segment in data.extra_fields.get("trajectory_segments", []) if segment["start"] < retained
        ]
        data.extra_fields.update({name: kwargs[name] for name in (
            "trajectory_id", "rollout_group_id", "rollout_sample_index", "rollout_group_size",
        )})
        data.extra_fields.update({
            "trajectory_execution_generation": generation,
            "completed_tool_batches": deepcopy(data.completed_tool_batches),
            "turn_scores": data.turn_scores, "tool_rewards": data.tool_rewards,
        })
        return AgentLoopOutput(
            prompt_ids=list(prompt), response_ids=response[:self.response_length], response_mask=mask,
            response_logprobs=logs, multi_modal_data={}, mm_processor_kwargs={}, mm_processor_output=None,
            num_turns=data.user_turns + data.assistant_turns + 1, metrics=data.metrics,
            routed_experts=None, extra_fields=data.extra_fields,
        )
