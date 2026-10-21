# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Pure observation adjudication; no Ray, model, tokenizer or optimizer load."""

import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[4] / "project-b-final-validation-20260928" / "production-entry-20261008" / "run_v1_functional_gate.py"
SPEC = importlib.util.spec_from_file_location("v1_functional_gate_observation_tests", SOURCE)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_manifest_actual_composition_is_not_the_driver_expectation():
    table = {"group_size": 8, "completed": [0, 1], "checkpointed": [2, 3, 4, 5], "pending": [6, 7]}
    assert gate.manifest_composition(table)["exact_2_4_2"]
    changed = {"group_size": 8, "completed": [0, 1, 2], "checkpointed": [3, 4, 5], "pending": [6, 7]}
    assert not gate.manifest_composition(changed)["exact_2_4_2"]


def test_acceptance_boundary_ready_requires_exact_member_identity_and_release_state():
    status = {
        "completed_members": [0, 1],
        "running_members": [2, 3, 4, 5],
        "pending_members": [6, 7],
        "starting_members": [],
        "acceptance_boundary_members": [2, 3, 4, 5],
        "acceptance_boundary_ready_members": [2, 3, 4, 5],
        "acceptance_boundary_released": False,
    }
    assert gate.acceptance_fixture_boundary_ready(status) is True
    assert gate.acceptance_fixture_boundary_ready({**status, "acceptance_boundary_ready_members": [2, 3, 4]}) is False
    assert gate.acceptance_fixture_boundary_ready({**status, "completed_members": [0, 2]}) is False
    assert gate.acceptance_fixture_boundary_ready({**status, "acceptance_boundary_released": True}) is False


def test_classify_acceptance_handoff_window_allows_opt_in_legitimate_three_three_two_only():
    status = {
        "completed_members": [0, 1, 4],
        "running_members": [2, 3, 5],
        "pending_members": [6, 7],
        "starting_members": [],
        "failed_members": [],
        "acceptance_boundary_members": [2, 3, 4, 5],
        "acceptance_boundary_ready_members": [2, 3, 5],
        "acceptance_boundary_released": False,
    }
    assert gate.classify_acceptance_handoff_window(status) is None
    observed = gate.classify_acceptance_handoff_window(status, allow_legitimate_source_completions=True)
    assert observed["completed"] == [0, 1, 4]
    assert observed["checkpointed"] == [2, 3, 5]
    assert observed["pending"] == [6, 7]
    assert observed["strict_exact_2_4_2"] is False
    assert sorted(observed["completed"] + observed["checkpointed"] + observed["pending"]) == list(range(8))


@pytest.mark.parametrize("status", [
    {"completed_members": [0, 1, 4], "running_members": [2, 3], "pending_members": [6, 7],
     "starting_members": [], "failed_members": [], "acceptance_boundary_members": [2, 3, 4, 5],
     "acceptance_boundary_ready_members": [2, 3], "acceptance_boundary_released": False},
    {"completed_members": [0, 1, 4], "running_members": [], "pending_members": [6, 7],
     "starting_members": [], "failed_members": [], "acceptance_boundary_members": [2, 3, 4, 5],
     "acceptance_boundary_ready_members": [], "acceptance_boundary_released": False},
    {"completed_members": [0, 1, 4], "running_members": [2, 3, 5], "pending_members": [6, 7],
     "starting_members": [], "failed_members": [], "acceptance_boundary_members": [2, 3, 4, 5],
     "acceptance_boundary_ready_members": [2, 3], "acceptance_boundary_released": False},
])
def test_classify_acceptance_handoff_window_rejects_invalid_legitimate_mix(status):
    assert gate.classify_acceptance_handoff_window(status, allow_legitimate_source_completions=True) is None


def test_should_break_source_window_poll_keeps_waiting_for_valid_opt_in_three_three_two_progress():
    status = {
        "completed_members": [0, 1, 4],
        "running_members": [2, 3, 5],
        "pending_members": [6, 7],
        "starting_members": [],
        "failed_members": [],
        "acceptance_boundary_ready_members": [2, 3],
    }
    assert gate.should_break_source_window_poll(status, allow_legitimate_source_completions=False) is True
    assert gate.should_break_source_window_poll(status, allow_legitimate_source_completions=True) is False


def test_should_break_source_window_poll_rejects_invalid_opt_in_terminal_states():
    assert gate.should_break_source_window_poll(
        {"completed_members": [0, 1, 4, 5], "running_members": [2, 3], "pending_members": [6, 7],
         "starting_members": [], "failed_members": [], "acceptance_boundary_ready_members": [2, 3]},
        allow_legitimate_source_completions=True,
    ) is True
    assert gate.should_break_source_window_poll(
        {"completed_members": [0, 1, 4], "running_members": [2, 3, 5], "pending_members": [6, 7],
         "starting_members": [], "failed_members": [5], "acceptance_boundary_ready_members": [2, 3, 5]},
        allow_legitimate_source_completions=True,
    ) is True


def test_should_break_source_window_poll_does_not_stop_during_initial_admission():
    status = {
        "completed_members": [], "running_members": [], "pending_members": list(range(8)),
        "starting_members": list(range(6)), "failed_members": [],
        "acceptance_boundary_members": [2, 3, 4, 5],
        "acceptance_boundary_ready_members": [], "acceptance_boundary_released": False,
    }
    assert gate.should_break_source_window_poll(status, allow_legitimate_source_completions=True) is False
    assert gate.classify_acceptance_handoff_window(status, allow_legitimate_source_completions=True) is None


def test_exact_acceptance_manifest_members_requires_expected_indices():
    class Manifest:
        completed_members = [{"member_index": 0}, {"member_index": 1}]
        checkpointed_trajectories = [
            {"member_index": 2}, {"member_index": 3}, {"member_index": 4}, {"member_index": 5}
        ]
        pending_samples = [{"member_index": 6}, {"member_index": 7}]

    exact = gate.exact_acceptance_manifest_members(Manifest())
    assert exact["exact_members"] is True
    wrong = Manifest()
    wrong.completed_members = [{"member_index": 0}, {"member_index": 2}]
    assert gate.exact_acceptance_manifest_members(wrong)["exact_members"] is False


def test_exact_acceptance_manifest_members_supports_actual_observed_mix():
    class Manifest:
        completed_members = [{"member_index": 0}, {"member_index": 1}, {"member_index": 4}]
        checkpointed_trajectories = [{"member_index": 2}, {"member_index": 3}, {"member_index": 5}]
        pending_samples = [{"member_index": 6}, {"member_index": 7}]

    exact = gate.exact_acceptance_manifest_members(
        Manifest(), expected_window={"completed": [0, 1, 4], "checkpointed": [2, 3, 5], "pending": [6, 7]}
    )
    assert exact["exact_members"] is True


def test_interpret_gate_status_keeps_overall_partial_when_strict_case_missing():
    assert gate.interpret_gate_status(strict_exact_2_4_2=True, mainline_ok=True) == {
        "strict_handoff_case_status": "PASS", "mainline_functional_status": "PASS", "status": "PASS"
    }
    assert gate.interpret_gate_status(strict_exact_2_4_2=False, mainline_ok=True) == {
        "strict_handoff_case_status": "PARTIAL", "mainline_functional_status": "PASS", "status": "PARTIAL"
    }


def test_build_control_accepts_default_false_and_enabled_true_without_numeric_failure():
    args = type("Args", (), {
        "observe_timeout_s": 90.0,
        "source_settle_timeout_s": 300.0,
        "restart_timeout_s": 120.0,
        "rpc_timeout_s": 30.0,
        "poll_interval_s": 0.1,
        "allow_legitimate_source_completions": False,
    })()
    control = gate.build_control(args)
    assert control["allow_legitimate_source_completions"] is False
    args.allow_legitimate_source_completions = True
    control = gate.build_control(args)
    assert control["allow_legitimate_source_completions"] is True


def test_initialize_driver_step_profile_state_matches_original_fit_loop():
    from types import SimpleNamespace

    trainer = SimpleNamespace(
        global_steps=3,
        config=SimpleNamespace(global_profiler=SimpleNamespace(steps=[3, 5])),
    )
    gate.initialize_driver_step_profile_state(trainer)
    assert trainer.prev_step_profile is False
    assert trainer.curr_step_profile is True
    assert trainer.next_step_profile is False

    trainer.global_steps = 4
    trainer.config.global_profiler.steps = None
    gate.initialize_driver_step_profile_state(trainer)
    assert trainer.prev_step_profile is False
    assert trainer.curr_step_profile is False
    assert trainer.next_step_profile is False


def test_initialize_driver_step_profile_state_supports_actual_separate_async_on_step_end_lifecycle():
    from types import SimpleNamespace
    from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync

    trainer = object.__new__(PPOTrainerSeparateAsync)
    trainer.global_steps = 1
    trainer.config = SimpleNamespace(global_profiler=SimpleNamespace(steps=None, profile_continuous_steps=False))
    trainer.hybrid_rollout_config = SimpleNamespace(enable_switch=False)
    trainer.timing_raw = {}
    trainer._pending_sync_metrics = {}
    trainer.standalone_checkpoint_manager = SimpleNamespace(update_weights=lambda step: {"updated_step": step})
    trainer._stop_rollout_profiling = lambda: (_ for _ in ()).throw(AssertionError("must not profile"))

    gate.initialize_driver_step_profile_state(trainer)
    PPOTrainerSeparateAsync.on_step_end(trainer)

    assert trainer.prev_step_profile is False
    assert trainer.curr_step_profile is False
    assert trainer.next_step_profile is False
    assert trainer._pending_sync_metrics == {"updated_step": 1}


def test_tool_counts_need_root_wiring_only_flags_missing_actual_counts_for_observed_tool_calls():
    assert gate.tool_counts_need_root_wiring([
        {"expected_search_calls_from_batches": 1, "expected_detail_calls_from_batches": 0,
         "actual_search_calls": None, "actual_detail_ids": None}
    ]) is True
    assert gate.tool_counts_need_root_wiring([
        {"expected_search_calls_from_batches": 0, "expected_detail_calls_from_batches": 0,
         "actual_search_calls": None, "actual_detail_ids": None}
    ]) is False
    assert gate.tool_counts_need_root_wiring([
        {"expected_search_calls_from_batches": 1, "expected_detail_calls_from_batches": 1,
         "actual_search_calls": 1, "actual_detail_ids": ["V001"]}
    ]) is False


def test_verify_active_published_policy_record_checks_identity_and_engine_observations():
    record = {
        "run_id": "fixture-run",
        "identity": {
            "policy_version": "7",
            "model_weights_fingerprint": "weights",
            "tokenizer_template_fingerprint": "tokenizer",
            "kv_cache_schema_fingerprint": "kv",
        },
        "model_configuration_fingerprint": "model-config",
        "publisher_id": "publisher",
        "engine_observations": [
            {"engine_url": "http://engine-a", "weight_version": "7", "source": "model_info"},
            {"engine_url": "http://engine-b", "weight_version": "7", "source": "model_info"},
        ],
        "committed_at_unix": 1.0,
    }
    verified = gate.verify_active_published_policy_record(
        record,
        expected_identity=record["identity"],
        expected_version=7,
        expected_registrations=[
            {"url": "http://engine-b", "engine_weight_version": "7"},
            {"url": "http://engine-a", "engine_weight_version": "7"},
        ],
    )
    assert verified["policy_version"] == "7"
    assert verified["engine_observations"][0]["engine_url"] == "http://engine-a"
    with pytest.raises(ValueError):
        gate.verify_active_published_policy_record(record, expected_identity={**record["identity"], "policy_version": "8"},
                                                   expected_version=7)


def test_run_continuous_policy_publication_observation_disabled_returns_none():
    result = gate.run_continuous_policy_publication_observation(
        enabled=False,
        trainer=object(),
        settings=type("Settings", (), {"run_id": "fixture-run"})(),
        base_identity={
            "policy_version": "1",
            "model_weights_fingerprint": "weights",
            "tokenizer_template_fingerprint": "tokenizer",
            "kv_cache_schema_fingerprint": "kv",
        },
        get_active_published_policy=lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
        prompt_ids=[1, 2, 3],
        request_namespace="fixture",
    )
    assert result is None


def test_run_continuous_policy_publication_observation_verifies_publication_and_generation():
    class FakeOutput:
        def __init__(self, extra_fields, *, token_ids, log_probs):
            self.extra_fields = extra_fields
            self.token_ids = token_ids
            self.log_probs = log_probs

    class FakeLLMClient:
        def __init__(self, output):
            self._output = output

        async def generate(self, request_id, *, prompt_ids, sampling_params, cache_salt=None):
            assert request_id.startswith("fixture/continuous-policy-publication/")
            assert prompt_ids == [11, 12, 13]
            assert sampling_params == {"max_new_tokens": 1, "temperature": 0, "logprobs": True}
            assert cache_salt == self._output.extra_fields["native_gateway_attempts"][-1]["cache_salt"]
            return self._output

    class FakeTrainer:
        def __init__(self, output):
            self.global_steps = 2
            self._output = output
            self.on_step_end_calls = 0

        def on_step_end(self):
            self.on_step_end_calls += 1
            self.global_steps = 3

        def get_llm_client(self):
            return FakeLLMClient(self._output)

    base_identity = {
        "policy_version": "2",
        "model_weights_fingerprint": "weights",
        "tokenizer_template_fingerprint": "tokenizer",
        "kv_cache_schema_fingerprint": "kv",
    }
    expected_identity = {**base_identity, "policy_version": "3"}
    from verl.trainer.ppo.v1.trajectory_state.identity import CanonicalPolicyKVIdentity
    from verl.workers.rollout.sglang_gateway_client import SGLangGatewayClient
    cache_salt = SGLangGatewayClient.build_policy_cache_salt(
        run_namespace="fixture-run",
        policy_version="3",
        policy_identity_salt=CanonicalPolicyKVIdentity(**expected_identity).cache_salt,
    )
    events = []
    trainer = FakeTrainer(FakeOutput({
        "global_steps": 3,
        "native_gateway_attempts": [{
            "output_start": 0,
            "output_end": 1,
            "cache_salt": cache_salt,
            "engine_weight_version": "3",
            "global_steps": 3,
            "num_cached_tokens": 5,
            "native_meta_info": {"id": "native-internal-request-id"},
        }],
    }, token_ids=[42], log_probs=[-0.5]))
    record = {
        "run_id": "fixture-run",
        "identity": expected_identity,
        "model_configuration_fingerprint": "model-config",
        "publisher_id": "publisher",
        "engine_observations": [{"engine_url": "http://engine-a", "weight_version": "3", "source": "model_info"}],
        "committed_at_unix": 1.0,
    }
    result = gate.run_continuous_policy_publication_observation(
        enabled=True,
        trainer=trainer,
        settings=type("Settings", (), {"run_id": "fixture-run"})(),
        base_identity=base_identity,
        get_active_published_policy=lambda: record,
        prompt_ids=[11, 12, 13],
        request_namespace="fixture",
        record_observation=lambda phase, **payload: events.append((phase, payload)),
    )
    assert trainer.on_step_end_calls == 1
    assert result["publication_verified"] is True
    assert result["generation_observation_verified"] is True
    assert result["active_published_policy_identity"] == expected_identity
    assert result["generation_observation"]["cache_salt"] == cache_salt
    assert result["generation_observation"]["native_gateway_attempt_count"] == 1
    assert result["generation_observation"]["native_gateway_attempt"]["native_request_id"] == "native-internal-request-id"
    assert [phase for phase, _payload in events] == [
        "continuous_policy_publication_verified", "continuous_policy_generation_receipt"
    ]


def test_observe_policy_cache_generation_rejects_missing_native_gateway_attempts():
    class FakeOutput:
        token_ids = [1]
        log_probs = [-0.1]
        extra_fields = {"global_steps": 3}

    class FakeLLMClient:
        async def generate(self, request_id, *, prompt_ids, sampling_params, cache_salt=None):
            return FakeOutput()

    with pytest.raises(ValueError, match="native_gateway_attempts"):
        gate.observe_policy_cache_generation(
            llm_client=FakeLLMClient(),
            settings=type("Settings", (), {"run_id": "fixture-run"})(),
            active_identity={
                "policy_version": "3",
                "model_weights_fingerprint": "weights",
                "tokenizer_template_fingerprint": "tokenizer",
                "kv_cache_schema_fingerprint": "kv",
            },
            prompt_ids=[11, 12],
            expected_global_steps=3,
            request_namespace="fixture",
        )


@pytest.mark.parametrize("attempt_patch,error_match", [
    ({"cache_salt": "wrong"}, "cache_salt differs"),
    ({"engine_weight_version": "4"}, "engine_weight_version differs"),
])
def test_observe_policy_cache_generation_rejects_bad_attempt_identity(attempt_patch, error_match):
    from verl.trainer.ppo.v1.trajectory_state.identity import CanonicalPolicyKVIdentity
    from verl.workers.rollout.sglang_gateway_client import SGLangGatewayClient

    identity = {
        "policy_version": "3",
        "model_weights_fingerprint": "weights",
        "tokenizer_template_fingerprint": "tokenizer",
        "kv_cache_schema_fingerprint": "kv",
    }
    cache_salt = SGLangGatewayClient.build_policy_cache_salt(
        run_namespace="fixture-run", policy_version="3",
        policy_identity_salt=CanonicalPolicyKVIdentity(**identity).cache_salt,
    )

    class FakeOutput:
        token_ids = [7]
        log_probs = [-0.3]
        extra_fields = {"native_gateway_attempts": [{
            "output_start": 0, "output_end": 1, "cache_salt": cache_salt,
            "engine_weight_version": "3", "global_steps": 3,
            "native_meta_info": {"id": "native-internal-request-id"},
        } | attempt_patch]}

    class FakeLLMClient:
        async def generate(self, request_id, *, prompt_ids, sampling_params, cache_salt=None):
            return FakeOutput()

    with pytest.raises(ValueError, match=error_match):
        gate.observe_policy_cache_generation(
            llm_client=FakeLLMClient(), settings=type("Settings", (), {"run_id": "fixture-run"})(),
            active_identity=identity, prompt_ids=[11, 12], expected_global_steps=3, request_namespace="fixture",
        )


def test_observe_policy_cache_generation_rejects_bad_attempt_spans_and_logprobs():
    from verl.trainer.ppo.v1.trajectory_state.identity import CanonicalPolicyKVIdentity
    from verl.workers.rollout.sglang_gateway_client import SGLangGatewayClient

    identity = {
        "policy_version": "3",
        "model_weights_fingerprint": "weights",
        "tokenizer_template_fingerprint": "tokenizer",
        "kv_cache_schema_fingerprint": "kv",
    }
    cache_salt = SGLangGatewayClient.build_policy_cache_salt(
        run_namespace="fixture-run", policy_version="3",
        policy_identity_salt=CanonicalPolicyKVIdentity(**identity).cache_salt,
    )

    class FakeOutput:
        def __init__(self, *, token_ids, log_probs, attempt):
            self.token_ids = token_ids
            self.log_probs = log_probs
            self.extra_fields = {"native_gateway_attempts": [attempt]}

    class FakeLLMClient:
        def __init__(self, output):
            self.output = output

        async def generate(self, request_id, *, prompt_ids, sampling_params, cache_salt=None):
            return self.output

    bad_span = {
        "output_start": 1, "output_end": 1, "cache_salt": cache_salt,
        "engine_weight_version": "3", "global_steps": 3,
        "native_meta_info": {"id": "native-internal-request-id"},
    }
    with pytest.raises(ValueError, match="output spans"):
        gate.observe_policy_cache_generation(
            llm_client=FakeLLMClient(FakeOutput(token_ids=[1], log_probs=[-0.1], attempt=bad_span)),
            settings=type("Settings", (), {"run_id": "fixture-run"})(),
            active_identity=identity, prompt_ids=[11], expected_global_steps=3, request_namespace="fixture",
        )

    good_span = {
        "output_start": 0, "output_end": 1, "cache_salt": cache_salt,
        "engine_weight_version": "3", "global_steps": 3,
        "native_meta_info": {"id": "native-internal-request-id"},
    }
    with pytest.raises(ValueError, match="log_probs"):
        gate.observe_policy_cache_generation(
            llm_client=FakeLLMClient(FakeOutput(token_ids=[1], log_probs=[float("nan")], attempt=good_span)),
            settings=type("Settings", (), {"run_id": "fixture-run"})(),
            active_identity=identity, prompt_ids=[11], expected_global_steps=3, request_namespace="fixture",
        )


@pytest.mark.parametrize("first_salt_matches", [True, False])
def test_generation_observation_checks_empty_attempt_before_real_token(first_salt_matches):
    from verl.trainer.ppo.v1.trajectory_state.identity import CanonicalPolicyKVIdentity
    from verl.workers.rollout.sglang_gateway_client import SGLangGatewayClient

    identity = {"policy_version": "3", "model_weights_fingerprint": "weights",
                "tokenizer_template_fingerprint": "tokenizer", "kv_cache_schema_fingerprint": "kv"}
    salt = SGLangGatewayClient.build_policy_cache_salt(
        run_namespace="fixture-run", policy_version="3",
        policy_identity_salt=CanonicalPolicyKVIdentity(**identity).cache_salt,
    )
    common = {"cache_salt": salt, "engine_weight_version": "3", "global_steps": 3,
              "native_meta_info": {"id": "fixture/continuous-policy-publication/3"}}

    class FakeOutput:
        token_ids = [7]
        log_probs = [-0.3]
        extra_fields = {"native_gateway_attempts": [
            {**common, "output_start": 0, "output_end": 0,
             "cache_salt": salt if first_salt_matches else "wrong"},
            {**common, "output_start": 0, "output_end": 1},
        ]}

    class FakeLLMClient:
        async def generate(self, request_id, *, prompt_ids, sampling_params, cache_salt=None):
            return FakeOutput()

    kwargs = dict(llm_client=FakeLLMClient(), settings=type("Settings", (), {"run_id": "fixture-run"})(),
                  active_identity=identity, prompt_ids=[11], expected_global_steps=3, request_namespace="fixture")
    if first_salt_matches:
        assert gate.observe_policy_cache_generation(**kwargs)["native_gateway_attempt_count"] == 2
    else:
        with pytest.raises(ValueError, match="cache_salt differs"):
            gate.observe_policy_cache_generation(**kwargs)


def test_run_continuous_policy_publication_observation_persists_publication_before_generation_failure():
    class FakeOutput:
        token_ids = [1]
        log_probs = [-0.1]
        extra_fields = {"native_gateway_attempts": [{"output_start": 0, "output_end": 1}]}

    class FakeLLMClient:
        async def generate(self, request_id, *, prompt_ids, sampling_params, cache_salt=None):
            return FakeOutput()

    class FakeTrainer:
        def __init__(self):
            self.global_steps = 2

        def on_step_end(self):
            self.global_steps = 3

        def get_llm_client(self):
            return FakeLLMClient()

    events = []
    identity = {
        "policy_version": "2",
        "model_weights_fingerprint": "weights",
        "tokenizer_template_fingerprint": "tokenizer",
        "kv_cache_schema_fingerprint": "kv",
    }
    record = {
        "run_id": "fixture-run",
        "identity": {**identity, "policy_version": "3"},
        "model_configuration_fingerprint": "model-config",
        "publisher_id": "publisher",
        "engine_observations": [{"engine_url": "http://engine-a", "weight_version": "3", "source": "model_info"}],
        "committed_at_unix": 1.0,
    }
    with pytest.raises(ValueError):
        gate.run_continuous_policy_publication_observation(
            enabled=True,
            trainer=FakeTrainer(),
            settings=type("Settings", (), {"run_id": "fixture-run"})(),
            base_identity=identity,
            get_active_published_policy=lambda: record,
            prompt_ids=[11, 12],
            request_namespace="fixture",
            record_observation=lambda phase, **payload: events.append((phase, payload)),
        )
    assert events[0][0] == "continuous_policy_publication_verified"
    assert events[1][0] == "continuous_policy_generation_receipt"


def test_verify_policy_cache_generation_receipt_rejects_missing_or_empty_native_request_id():
    identity = {
        "policy_version": "3",
        "model_weights_fingerprint": "weights",
        "tokenizer_template_fingerprint": "tokenizer",
        "kv_cache_schema_fingerprint": "kv",
    }
    for native_meta_info in ({}, {"id": ""}, None):
        with pytest.raises(ValueError, match="native request id"):
            gate.verify_policy_cache_generation_receipt(
                request_id="outer-request",
                signature="(request_id, *, prompt_ids, sampling_params, cache_salt=None)",
                sampling_params={"max_new_tokens": 1, "temperature": 0, "logprobs": True},
                prompt_ids=[11],
                active_identity=identity,
                expected_global_steps=3,
                expected_cache_salt="salt",
                token_ids=[1],
                log_probs=[-0.1],
                extra_fields={"native_gateway_attempts": [{
                    "output_start": 0,
                    "output_end": 1,
                    "cache_salt": "salt",
                    "engine_weight_version": "3",
                    "global_steps": 3,
                    "native_meta_info": native_meta_info,
                }]},
            )


def test_unwrap_observed_value_prefers_real_torch_tensor_over_forwarded_wrapper_attributes():
    import torch

    class ForwardingWrapper:
        def __init__(self, wrapped):
            self.data = wrapped

        def __getattr__(self, name):
            return getattr(self.data, name)

    scalar = ForwardingWrapper(torch.tensor(7, dtype=torch.int64))
    vector = ForwardingWrapper(torch.tensor([1, 2, 3], dtype=torch.int64))

    assert gate._unwrap_observed_value(scalar) == 7
    assert gate._unwrap_observed_value(vector) == [1, 2, 3]


@pytest.mark.parametrize("pending", [[6, 6], [6], [6, 8]])
def test_missing_duplicate_foreign_members_are_not_accepted(pending):
    with pytest.raises(ValueError):
        gate.manifest_composition({"group_size": 8, "completed": [0, 1], "checkpointed": [2, 3, 4, 5], "pending": pending})


def test_actor_kill_intent_or_stale_state_is_not_restart_proof():
    before = {"actor_id": "actual-actor", "pid": 101, "node_id": "node-a", "num_restarts": 0}
    assert gate.actor_restart_observed(before, before) is None
    assert gate.actor_restart_observed(before, {**before, "num_restarts": 1}) is True
    assert gate.actor_restart_observed(before, {**before, "pid": 102}) is True
    assert gate.actor_restart_observed(before, {**before, "actor_id": "replacement-actor", "pid": 102}) is None
    assert gate.actor_restart_observed({}, {}) is None


def test_parameter_version_is_actual_checkpoint_publication_not_prompt_birth_step():
    declared = {"policy_version": gate.PENDING_VERSION, "model_weights_fingerprint": "actual-files"}
    actual = gate.bind_initialized_policy_version(
        declared, observed_version=7, checkpoint_step=7,
        registrations=[{"engine_weight_version": "7", "url": "http://actual-engine"}],
    )
    assert actual["policy_version"] == "7"
    assert declared["policy_version"] == gate.PENDING_VERSION
    assert actual["model_weights_fingerprint"] == "actual-files"
    with pytest.raises(ValueError):
        gate.bind_initialized_policy_version(declared, observed_version=8, checkpoint_step=7,
                                             registrations=[{"engine_weight_version": "8"}])
    with pytest.raises(ValueError):
        gate.bind_initialized_policy_version(declared, observed_version=7, checkpoint_step=7, registrations=[])
    with pytest.raises(ValueError):
        gate.bind_initialized_policy_version(declared, observed_version=7, checkpoint_step=7,
                                             registrations=[{"engine_weight_version": "6"}])


def actual_batches():
    return [
        {"turn_index": 0, "calls": [{"name": "search_candidates", "arguments": '{"district":"fixture"}'}],
         "results": [{"role": "tool", "content": '{"candidates":[]}'}]},
        {"turn_index": 1, "calls": [{"name": "query_venue_detail", "arguments": '{"venue_id":"V001"}'}],
         "results": [{"role": "tool", "content": '{"id":"V001"}'}]},
    ]


def test_actual_handler_counters_are_required_to_report_zero_tool_replay():
    batches = actual_batches()
    proof = gate.verify_local_life_tool_counts(batches, {"local_life_search_calls": 1, "local_life_detail_ids": ["V001"]})
    assert proof["tool_replay_count"] == 0
    assert gate.verify_local_life_tool_counts(batches, {})["tool_replay_count"] is None
    excess = gate.verify_local_life_tool_counts(batches, {"local_life_search_calls": 2, "local_life_detail_ids": ["V001", "V001"]})
    assert excess["tool_replay_count"] == 2


def test_committed_tool_batches_without_real_handler_counters_stay_null():
    batches = actual_batches()
    proof = gate.resolve_local_life_tool_evidence(completed_tool_batches=batches, extra_fields={})
    assert proof["tool_replay_count"] is None
    assert proof["reason"] == "prepared_extra_fields_missing_actual_local_life_counts"
    assert proof["expected_count_source"] == "committed_completed_tool_batches"
    assert proof["actual_count_source"] is None
    assert proof["prepared_extra_fields_search_calls"] is None
    assert proof["prepared_extra_fields_detail_ids"] is None


def test_resolve_local_life_tool_evidence_accepts_wrapped_integral_and_id_stack():
    import torch

    class Box:
        def __init__(self, value):
            self.value = value

    batches = actual_batches()
    proof = gate.resolve_local_life_tool_evidence(
        completed_tool_batches=batches,
        extra_fields={
            "local_life_search_calls": Box(torch.tensor(1, dtype=torch.int64)),
            "local_life_detail_ids": Box(["V001"]),
        },
    )
    assert proof["tool_replay_count"] == 0
    assert proof["actual_search_calls"] == 1
    assert proof["actual_detail_ids"] == ["V001"]
    assert proof["actual_count_source"] == "agent_data.extra_fields"


@pytest.mark.parametrize("searches", [True, 1.0, "1"])
def test_resolve_local_life_tool_evidence_rejects_non_integral_handler_counter(searches):
    proof = gate.resolve_local_life_tool_evidence(
        completed_tool_batches=actual_batches(),
        extra_fields={"local_life_search_calls": searches, "local_life_detail_ids": ["V001"]},
    )
    assert proof["tool_replay_count"] is None


@pytest.mark.parametrize("detail_ids", [[1], [1.0], [True], "V001"])
def test_resolve_local_life_tool_evidence_rejects_invalid_detail_id_container(detail_ids):
    proof = gate.resolve_local_life_tool_evidence(
        completed_tool_batches=actual_batches(),
        extra_fields={"local_life_search_calls": 1, "local_life_detail_ids": detail_ids},
    )
    assert proof["tool_replay_count"] is None


def test_unknown_tool_or_incomplete_counters_are_null_not_zero():
    batches = actual_batches()
    batches[0]["calls"][0]["name"] = "unknown_tool"
    assert gate.verify_local_life_tool_counts(batches, {})["tool_replay_count"] is None
    assert gate.verify_local_life_tool_counts(actual_batches(), {"local_life_search_calls": 0, "local_life_detail_ids": []})["tool_replay_count"] is None


def test_detail_error_does_not_count_as_successful_handler_record():
    batches = actual_batches()
    batches[1]["results"][0]["content"] = '{"error":"unknown venue_id"}'
    proof = gate.verify_local_life_tool_counts(batches, {"local_life_search_calls": 1, "local_life_detail_ids": []})
    assert proof["tool_replay_count"] == 0 and proof["detail_calls"] == 0


def parameter_rows(values, *, step=None, state_entries=0):
    return [{"rank": 0, "optimizer_step_max": step, "optimizer_step_min": step,
             "optimizer_state_entries": state_entries,
             "samples": [{"name": "actual-local-shard", "local_shape": [128], "dtype": "torch.float32",
                          "indices": [0, 127], "values": values}]}]


def test_real_parameter_samples_and_optimizer_steps_are_independent_proofs():
    before = parameter_rows([1.0, 2.0])
    after = parameter_rows([1.0, 2.125], step=1, state_entries=2)
    proof = gate.parameter_sample_delta(before, after)
    assert proof["observed_parameter_change"] is True and proof["sample_linf"] == 0.125
    assert proof["optimizer_step_advanced"] is True
    same = gate.parameter_sample_delta(before, parameter_rows([1.0, 2.0], step=1, state_entries=2))
    assert same["observed_parameter_change"] is False and same["optimizer_step_advanced"] is True


def test_missing_optimizer_counter_does_not_become_a_real_update():
    proof = gate.parameter_sample_delta(parameter_rows([1.0, 2.0]), parameter_rows([1.0, 2.125], state_entries=2))
    assert proof["observed_parameter_change"] is True
    assert proof["optimizer_step_advanced"] is None


def test_changed_parameter_partition_cannot_be_compared_as_update():
    before, after = parameter_rows([1.0, 2.0]), parameter_rows([1.0, 2.125], step=1, state_entries=2)
    after[0]["samples"][0]["indices"] = [1, 126]
    assert gate.parameter_sample_delta(before, after)["observed_parameter_change"] is None
    assert gate.parameter_sample_delta([], after)["observed_parameter_change"] is None


def test_parameter_probe_reads_the_original_actor_inside_worker_dict():
    from types import SimpleNamespace
    import torch

    module = torch.nn.Linear(3, 2)
    optimizer = SimpleNamespace(state={}, param_groups=[{"lr": 1e-6}])
    engine = SimpleNamespace(module=module, optimizer=optimizer)
    worker = SimpleNamespace(rank=0, worker_dict={"actor": SimpleNamespace(actor=SimpleNamespace(engine=engine))})
    before = {name: value.detach().clone() for name, value in module.named_parameters()}
    result = gate._parameter_probe(worker)
    assert result.get("error_type") is None
    assert result["rank"] == 0 and len(result["samples"]) == 2
    assert result["optimizer_state_entries"] == 0
    assert result["optimizer_learning_rates"] == [1e-6]
    assert all(torch.equal(before[name], value.detach()) for name, value in module.named_parameters())


def test_parameter_probe_does_not_use_an_unrelated_worker_role():
    from types import SimpleNamespace

    result = gate._parameter_probe(SimpleNamespace(rank=0, worker_dict={"ref": object()}))
    assert result["error_type"] == "KeyError" and result["samples"] == []


def test_uniform_sample_positions_cover_edges_without_float_rounding():
    assert gate._uniform_sample_positions(0) == []
    assert gate._uniform_sample_positions(1) == [0]
    assert gate._uniform_sample_positions(5) == [0, 1, 2, 3, 4]
    large = gate._uniform_sample_positions(136134656)
    assert len(large) == 64
    assert large[0] == 0
    assert large[-1] == 136134655
    assert all(isinstance(index, int) for index in large)
    assert all(0 <= index < 136134656 for index in large)
    assert all(left <= right for left, right in zip(large, large[1:]))



def test_validate_parameter_probe_indices_rejects_out_of_range_endpoints():
    with pytest.raises(ValueError):
        gate._validate_parameter_probe_indices([0, 5], 5)
    with pytest.raises(ValueError):
        gate._validate_parameter_probe_indices([], 3)



def test_parameter_probe_large_index_generation_stays_in_range_without_allocating_large_tensor(monkeypatch):
    from types import SimpleNamespace

    class FakeTensor:
        def __init__(self, values, *, device='cuda:0', dtype='torch.float32'):
            self._values = list(values)
            self.device = device
            self.dtype = dtype

        def detach(self):
            return self

        def reshape(self, *_args, **_kwargs):
            return self

        def numel(self):
            return 136134656

        @property
        def shape(self):
            return (4096, 33236)

        def __getitem__(self, indices):
            values = indices.tolist()
            assert values[0] == 0
            assert values[-1] == 136134655
            assert len(values) == 64
            assert all(0 <= index < 136134656 for index in values)
            return FakeTensor([float(index) for index in values], device='cpu', dtype='torch.float32')

        def float(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self._values)

    class FakeParameter(FakeTensor):
        requires_grad = True

    class FakeModule:
        def named_parameters(self):
            yield 'large.weight', FakeParameter([])

    fake_torch = SimpleNamespace(
        int64='int64',
        Tensor=type('FakeTorchTensor', (), {}),
        tensor=lambda values, *, device, dtype: FakeTensor(values, device=device, dtype=dtype),
    )
    monkeypatch.setitem(__import__('sys').modules, 'torch', fake_torch)
    optimizer = SimpleNamespace(state={}, param_groups=[{'lr': 1e-6}])
    worker = SimpleNamespace(rank=0, engine=SimpleNamespace(module=FakeModule(), optimizer=optimizer))

    result = gate._parameter_probe(worker)
    assert result.get('error_type') is None
    assert result['samples'][0]['indices'][-1] == 136134655



def test_record_pre_update_probe_or_raise_stops_before_training_update(tmp_path):
    evidence = gate.Evidence(tmp_path)
    before_parameters = [{'rank': 0, 'error_type': 'ValueError', 'error_message': 'bad index', 'samples': []}]

    with pytest.raises(ValueError, match='Parameter probe failed before the original training update'):
        gate._record_pre_update_probe_or_raise(evidence, before_parameters)

    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['phase'] == 'parameter_probe_before_update'
    assert result['parameter_probe_result'][0]['error_type'] == 'ValueError'
    assert result['optimizer_calls_this_driver'] is None


def test_archive_training_group_evidence_is_atomic_and_preserves_member_fields(tmp_path):
    import torch
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorData, NonTensorStack

    fields = TensorDict({
        'session_id': torch.tensor([0, 1], dtype=torch.int64),
        'responses': torch.nested.as_nested_tensor([
            torch.tensor([11, 12], dtype=torch.int64),
            torch.tensor([21, 22, 23], dtype=torch.int64),
        ], layout=torch.jagged),
        'response_mask': torch.nested.as_nested_tensor([
            torch.tensor([True, True]),
            torch.tensor([True, True, True]),
        ], layout=torch.jagged),
        'rollout_log_probs': torch.nested.as_nested_tensor([
            torch.tensor([0.1, 0.2], dtype=torch.float32),
            torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32),
        ], layout=torch.jagged),
        'extra_fields': NonTensorStack.from_list([
            NonTensorData({'local_life_search_calls': torch.tensor(1, dtype=torch.int64),
                           'local_life_detail_ids': ['V001']}),
            NonTensorData({'local_life_search_calls': torch.tensor(2, dtype=torch.int64),
                           'local_life_detail_ids': ['V002', 'V003']}),
        ]),
    }, batch_size=[2])
    archive = gate.archive_training_group_evidence(
        output_dir=tmp_path, uid='u0', keys=['u0_0_0', 'u0_1_0'], fields=fields,
        tags=[{'status': 'finished'}, {'status': 'finished'}], binding={'content_digest': 'abc'})
    saved = torch.load(archive['path'], weights_only=False)
    assert archive['byte_size'] > 0 and len(archive['sha256']) == 64
    assert saved['keys'] == ['u0_0_0', 'u0_1_0']
    assert saved['tags'][0]['status'] == 'finished'
    assert gate._unwrap_integral_counter(saved['fields']['session_id'][1]) == 1
    assert gate._mapping_keys(saved['fields']['extra_fields'][0]) == ['local_life_detail_ids', 'local_life_search_calls']
    assert gate._unwrap_string_list(gate._extract_mapping_value(saved['fields']['extra_fields'][1], 'local_life_detail_ids')) == ['V002', 'V003']
    assert any(item['dtype'] == 'torch.float32' for item in archive['members'][0]['tensor_members'])


def test_manifest_non_2_4_2_is_not_acceptable_for_final_pass():
    composition = gate.manifest_composition(
        {"group_size": 8, "completed": [0, 1, 2], "checkpointed": [3, 4, 5], "pending": [6, 7]}
    )
    assert composition["exact_2_4_2"] is False


def test_archive_failure_happens_before_optimizer_and_temp_file_is_not_promoted(tmp_path, monkeypatch):
    import torch
    from tensordict import TensorDict

    fields = TensorDict({'uid': ['u0'], 'session_id': torch.tensor([0], dtype=torch.int64)}, batch_size=[1])

    def boom(_payload, _stream):
        raise OSError('disk full')

    monkeypatch.setattr(torch, 'save', boom)
    with pytest.raises(OSError, match='disk full'):
        gate.archive_training_group_evidence(
            output_dir=tmp_path, uid='u0', keys=['u0_0_0'], fields=fields,
            tags=[{'status': 'finished'}], binding={'content_digest': 'abc'})
    assert not list((tmp_path / 'evidence').glob('training_group_u0.pt'))
    assert not list((tmp_path / 'evidence').glob('.training_group_u0.pt.tmp'))


def test_freeze_uses_actual_files_and_rejects_a_changed_original_input(tmp_path):
    config = tmp_path / "frozen.yaml"
    config.write_text("fixture: true\n")
    files = []
    for role in sorted(gate.REQUIRED_ROLES):
        path = tmp_path / (role + ".fixture")
        path.write_text("original fixture for " + role)
        files.append({"role": role, "path": str(path), "sha256": gate.file_sha256(path)})
    manifest = {"schema_version": 1, "data_kind": "synthetic/local_life",
                "parameter_identity_source": "explicit_frozen_manifest", "policy_identity": {"fixture": "identity"},
                "config_sha256": gate.file_sha256(config), "files": files}
    saved = tmp_path / "freeze.json"
    saved.write_text(json.dumps(manifest))
    assert gate.verify_freeze(config, saved) == manifest
    Path(files[0]["path"]).write_text("changed after freeze")
    with pytest.raises(ValueError):
        gate.verify_freeze(config, saved)


def test_missing_metric_is_not_imputed_and_nonfinite_metric_is_omitted():
    result = gate._finite_metrics({"actor/grad_norm": float("nan"), "actor/lr": 1e-6, "not_scalar": [1]})
    assert "actor/grad_norm" not in result and result["actor/lr"] == 1e-6


def test_evidence_defaults_keep_unobserved_fields_null(tmp_path):
    evidence = gate.Evidence(tmp_path)
    evidence.event("fixture_prepared")
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["stale_write_rejected"] is None
    assert result["non_zero_gradient"] is None and result["real_optimizer_update"] is None
    assert not result["old_gate_or_trainer_results_reused"]
    assert not result["queue_optimizer_ack_exactly_once_proved"]
    assert result["actual_parameter_version"] is None and result["prompt_birth_global_step"] is None
    assert result["actual_fields_archive"] is None
    assert result["manifest_window_driver_controlled"] is None
    assert result["optimizer_attempts_this_driver"] is None
    assert result["optimizer_completions_this_driver"] is None


def original_config():
    policy = {"policy_version": "0", "model_weights_fingerprint": "explicit-actual-weight-identity",
              "tokenizer_template_fingerprint": "explicit-actual-template-identity",
              "kv_cache_schema_fingerprint": "explicit-actual-layout-identity"}
    return {
        "actor_rollout_ref": {"hybrid_engine": False, "model": {"path": "/fixture/model"},
                             "actor": {"ppo_mini_batch_size": 1, "ppo_epochs": 1},
                             "rollout": {"n": 8, "calculate_log_probs": True,
                                         "trajectory_state": {"enabled": True, "coordinator_max_restarts": 1,
                                                               "policy_identity": policy},
                                         "agent": {"num_workers": 2, "default_agent_loop": "tool_agent"},
                                         "multi_turn": {"enable": True, "tool_config_path": "/fixture/tools"}}},
        "trainer": {"use_v1": True, "resume_mode": "disable",
                    "v1": {"trainer_mode": "separate_async", "separate_async": {"parameter_sync_step": 1},
                           "sampler": {"custom_sampler": None}}},
        "transfer_queue": {"enable": True}, "algorithm": {"adv_estimator": "grpo", "filter_groups": None},
        "data": {"train_batch_size": 1, "train_files": ["/fixture/original-data"]},
        "reward": {"custom_reward_function": {"path": "/fixture/original-reward", "name": "compute_score"}},
    }


def config_freeze(config):
    return {"policy_identity": config["actor_rollout_ref"]["rollout"]["trajectory_state"]["policy_identity"],
            "model_path": "/fixture/model", "files": [
                {"role": "train_data", "path": "/fixture/original-data"},
                {"role": "tools_config", "path": "/fixture/tools"},
                {"role": "reward_implementation", "path": "/fixture/original-reward"},
            ]}


def test_original_v1_config_allows_null_optional_sampler_and_filter():
    config = original_config()
    assert gate.validate_config(config, config_freeze(config))["enabled"]


@pytest.mark.parametrize("field,value", [("n", 7), ("calculate_log_probs", False), ("topk_log_probs", True)])
def test_gate_rejects_unverifiable_behavior_or_wrong_group_before_model_init(field, value):
    config = original_config()
    config["actor_rollout_ref"]["rollout"][field] = value
    with pytest.raises(ValueError):
        gate.validate_config(config, config_freeze(config))


def test_original_reward_and_data_must_match_the_frozen_source():
    config = original_config()
    frozen = config_freeze(config)
    config["reward"]["custom_reward_function"]["path"] = "/fixture/altered-reward"
    with pytest.raises(ValueError):
        gate.validate_config(config, frozen)
