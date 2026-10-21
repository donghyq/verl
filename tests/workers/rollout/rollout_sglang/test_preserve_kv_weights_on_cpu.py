# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Real adapter methods with CPU transport/collective collaborators, not a GPU test.

This public entry imports the actual adapter and native request/serializer modules.
IPC/GPU transfer and TP collectives are replaced only at their external boundaries.
"""

import asyncio
import base64
import copy
import importlib.util
import pickle
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, patch

if importlib.util.find_spec("sglang") is None:
    raise unittest.SkipTest("Requires SGLang; no adapter-module import or GPU verification performed")

import torch.distributed as dist
from sglang.srt import managers
from sglang.srt.utils import MultiprocessingSerializer, patch_torch

from verl.workers.rollout.sglang_rollout import sglang_rollout as rollout


class TensorValue:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self


class Mesh:
    def __init__(self, rank=0):
        self.rank = rank
        self.mesh = SimpleNamespace(size=lambda: (2,), tolist=lambda: [4, 5])

    def get_local_rank(self):
        return self.rank

    def get_group(self):
        return "test-infer-tp-group"


class TestPreservedWeightUpdate(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.mesh = Mesh()
        self.adapter = rollout.ServerAdapter.__new__(rollout.ServerAdapter)
        self.adapter.device_mesh = {"infer_tp": self.mesh}
        self.adapter._pd_role = None
        self.adapter.config = SimpleNamespace(
            checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1, backend="naive"),
            quantization=SimpleNamespace(enabled=False),
            engine_kwargs={},
            data_parallel_size=1,
            pipeline_model_parallel_size=1,
            disaggregation=SimpleNamespace(enabled=False),
            mtp=SimpleNamespace(enable=False, enable_rollout=False),
        )
        self.adapter.model_config = SimpleNamespace(
            lora_rank=0, lora={}, hf_config=SimpleNamespace(model_type="test", quantization_config=None)
        )
        self.adapter._init_server_adapter = AsyncMock()
        self.publish = AsyncMock(side_effect=lambda step: self.events.append(("publish", step)))
        self.adapter.server_actor = SimpleNamespace(set_global_steps=SimpleNamespace(remote=self.publish))
        self.transport = AsyncMock(side_effect=self.request)
        self.engine = SimpleNamespace(
            _make_async_request=self.transport,
            flush_cache=AsyncMock(side_effect=lambda: self.events.append(("flush",))),
            resume_memory_occupation=AsyncMock(),
        )
        self.adapter._engine = self.engine
        self.gathers = []
        self.preprocessed = []

        async def buckets(weights, _bucket_bytes, fusion_groups=()):
            for item in weights:
                yield [item]

        def gather(*, obj, object_gather_list, dst, group):
            self.gathers.append((obj, dst, group))
            if object_gather_list is not None:
                object_gather_list[:] = [
                    obj,
                    [(name, pickle.dumps(TensorValue("peer:" + pickle.loads(value).value))) for name, value in obj],
                ]

        def preprocess(tensor):
            self.preprocessed.append(tensor.value)
            return tensor

        for context in [
            patch.object(rollout, "_preserved_weight_update_uses_session", return_value=True),
            patch.object(rollout, "get_named_tensor_buckets", buckets),
            patch.object(rollout, "_preprocess_tensor_for_update_weights", preprocess),
            patch.object(rollout, "_to_ipc_device", lambda tensor: tensor),
            patch.object(MultiprocessingSerializer, "serialize", side_effect=pickle.dumps),
            patch.object(patch_torch, "monkey_patch_torch_reductions"),
            patch.object(dist, "gather_object", side_effect=gather),
            patch.object(dist, "broadcast_object_list", side_effect=self.broadcast),
        ]:
            context.start()
            self.addCleanup(context.stop)

    async def request(self, endpoint, payload):
        self.events.append((endpoint, payload))
        return {"success": True, "message": "Success"}

    @staticmethod
    def broadcast(outcome, *, src, group):
        # Single-participant tests supply the other rank's successful control vote.
        if outcome[0] is None:
            outcome[0] = True

    @staticmethod
    def weights():
        return iter([("layer.a", TensorValue("A")), ("layer.b", TensorValue("B"))])

    def run_update(self, weights=None, **kwargs):
        return asyncio.run(self.adapter.update_weights(self.weights() if weights is None else weights, **kwargs))

    def assert_no_publication_or_cleanup(self):
        self.publish.assert_not_awaited()
        self.engine.flush_cache.assert_not_awaited()
        self.engine.resume_memory_occupation.assert_not_awaited()

    def test_preserves_every_bucket_and_publishes_only_after_session_end(self):
        self.run_update(global_steps=12, preserve_kv_cache=True)
        self.assertEqual(
            [event[0] for event in self.events],
            [
                "begin_weight_update",
                "update_weights_from_tensor",
                "update_weights_from_tensor",
                "end_weight_update",
                "publish",
            ],
        )
        self.assertEqual(self.events[0][1], {"selector": "all"})
        self.assertEqual(self.events[-2][1], {})
        decoded = []
        for endpoint, payload in self.events[1:3]:
            self.assertEqual(endpoint, "update_weights_from_tensor")
            self.assertIs(payload["flush_cache"], False)
            self.assertIs(payload["abort_all_requests"], False)
            self.assertIsNone(payload["load_format"])
            self.assertEqual(len(payload["serialized_named_tensors"]), 2)
            tensors = pickle.loads(base64.b64decode(payload["serialized_named_tensors"][0]))
            decoded.extend((name, [pickle.loads(v).value for v in logical.values]) for name, logical in tensors)
        self.assertEqual(decoded, [("layer.a", ["A", "peer:A"]), ("layer.b", ["B", "peer:B"])])
        self.assertEqual(self.preprocessed, ["A", "B"])
        self.assertEqual([(dst, group) for _, dst, group in self.gathers], [(4, "test-infer-tp-group")] * 2)
        self.engine.flush_cache.assert_not_awaited()
        self.engine.resume_memory_occupation.assert_not_awaited()

    def test_nonleader_consumes_preprocesses_and_gathers_without_http(self):
        self.mesh.rank = 1
        self.adapter._engine = None
        self.run_update(global_steps=12, preserve_kv_cache=True)
        self.assertEqual(self.preprocessed, ["A", "B"])
        self.assertEqual(len(self.gathers), 2)
        self.transport.assert_not_awaited()
        self.publish.assert_not_awaited()

    def test_default_keeps_original_bucket_helper_flush_and_publication(self):
        helper = AsyncMock()
        with patch.object(rollout, "sgl_update_weights", helper):
            self.run_update(global_steps=12)
        self.assertEqual(helper.await_count, 2)
        self.transport.assert_not_awaited()
        self.assertEqual(self.events, [("flush",), ("publish", 12)])

    def test_default_delta_keeps_original_dispatch(self):
        helper = AsyncMock()
        self.adapter._update_weights_delta = helper
        weights = self.weights()
        self.run_update(weights, wire_format="delta_flush", global_steps=12)
        helper.assert_awaited_once_with(weights, global_steps=12)
        self.assert_no_publication_or_cleanup()

    def test_rejects_unsupported_modes_before_init_or_weight_consumption(self):
        cases = [
            ({"wire_format": "delta_flush"}, lambda: None),
            ({"peft_config": {}}, lambda: None),
            ({}, lambda: setattr(self.adapter.model_config, "lora_rank", 8)),
            ({}, lambda: setattr(self.adapter.model_config, "lora", {"rank": 8, "merge": True})),
            ({}, lambda: setattr(self.adapter.config, "quantization", SimpleNamespace(enabled=True))),
            ({}, lambda: setattr(self.adapter.config, "quantization", {"linear": "FP8"})),
            ({}, lambda: setattr(self.adapter.model_config.hf_config, "quantization_config", {"method": "fp8"})),
            ({}, lambda: setattr(self.adapter.config, "engine_kwargs", {"sglang": {"quantization": "fp8"}})),
            ({}, lambda: setattr(self.adapter.config, "data_parallel_size", 2)),
            ({}, lambda: setattr(self.adapter.config, "pipeline_model_parallel_size", 2)),
            ({}, lambda: setattr(self.adapter.config, "disaggregation", {"enabled": True})),
            ({}, lambda: setattr(self.adapter.config, "engine_kwargs", {"sglang": {"speculative_algorithm": "EAGLE"}})),
            ({}, lambda: setattr(self.adapter.config, "mtp", SimpleNamespace(enable=True, enable_rollout=True))),
            ({}, lambda: setattr(self.adapter.config.checkpoint_engine, "backend", "delta")),
        ]
        config, model = copy.deepcopy(self.adapter.config), copy.deepcopy(self.adapter.model_config)
        for index, (kwargs, mutate) in enumerate(cases):
            with self.subTest(index=index):
                self.adapter.config, self.adapter.model_config = copy.deepcopy(config), copy.deepcopy(model)
                mutate()

                def forbidden():
                    raise AssertionError("unsupported update consumed tensors")
                    yield

                with self.assertRaises(ValueError):
                    self.run_update(forbidden(), preserve_kv_cache=True, **kwargs)
                self.adapter._init_server_adapter.assert_not_awaited()
                self.transport.assert_not_awaited()
                self.assert_no_publication_or_cleanup()

    def test_begin_bucket_end_rejections_never_publish_flush_or_resume(self):
        for failure in ("begin_weight_update", "update_weights_from_tensor", "end_weight_update"):
            for response in ({"success": False, "message": "native rejected"}, {"message": "missing success"}):
                with self.subTest(endpoint=failure, response=response):
                    self.events.clear()

                    async def reject(endpoint, payload, response=response, failure=failure):
                        self.events.append((endpoint, payload))
                        return response if endpoint == failure else {"success": True}

                    self.transport.side_effect = reject
                    with self.assertRaisesRegex(RuntimeError, failure):
                        self.run_update(global_steps=12, preserve_kv_cache=True)
                    self.assert_no_publication_or_cleanup()
                    if failure != "end_weight_update":
                        self.assertNotIn("end_weight_update", [event[0] for event in self.events])

    def test_transport_failure_keeps_original_exception_and_no_cleanup(self):
        for error in (OSError("network transport failed"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                self.transport.side_effect = error
                with self.assertRaises(RuntimeError) as caught:
                    self.run_update(global_steps=12, preserve_kv_cache=True)
                self.assertIs(caught.exception.__cause__, error)
                self.assert_no_publication_or_cleanup()

    def test_empty_or_broken_weight_stream_never_finishes_or_publishes(self):
        def broken():
            yield "layer.a", TensorValue("A")
            raise RuntimeError("weight source failed")

        for weights, exception in [(iter([]), ValueError), (broken(), RuntimeError)]:
            with self.subTest(exception=exception):
                self.events.clear()
                with self.assertRaises(exception):
                    self.run_update(weights, global_steps=12, preserve_kv_cache=True)
                self.assertNotIn("end_weight_update", [event[0] for event in self.events])
                self.assert_no_publication_or_cleanup()

    def test_two_tp_participants_stop_before_next_gather_on_control_failure(self):
        """Actual methods, two threads; only the collective transport is a collaborator."""
        for failure, expected_gathers in [("begin_weight_update", 0), ("second_bucket", 2), ("end_weight_update", 3)]:
            with self.subTest(failure=failure):
                self._check_tp_control_failure(failure, expected_gathers)

    def _check_tp_control_failure(self, failure, expected_gathers):
        self.events.clear()
        barrier = threading.Barrier(2, timeout=3)
        local = threading.local()
        gathered = {}
        control = {}
        gather_counts = [0, 0]
        bucket_calls = 0

        def gather(*, obj, object_gather_list, dst, group):
            rank = local.rank
            gather_counts[rank] += 1
            gathered[rank] = obj
            barrier.wait()
            if rank == 0:
                object_gather_list[:] = [gathered[0], gathered[1]]
            barrier.wait()

        def broadcast(outcome, *, src, group):
            rank = local.rank
            if rank == 0:
                control["value"] = outcome[0]
            barrier.wait()
            outcome[0] = control["value"]
            barrier.wait()

        async def reject(endpoint, payload):
            nonlocal bucket_calls
            self.events.append((endpoint, payload))
            if endpoint == "update_weights_from_tensor":
                bucket_calls += 1
            rejected = endpoint == failure or (failure == "second_bucket" and bucket_calls == 2)
            return {"success": not rejected, "message": "/private/native/detail must not be echoed"}

        def participant(rank):
            local.rank = rank
            adapter = copy.copy(self.adapter)
            adapter.device_mesh = {"infer_tp": Mesh(rank)}
            adapter._engine = self.engine if rank == 0 else None
            weights = iter([(name, TensorValue(name)) for name in ("a", "b", "c")])
            try:
                asyncio.run(adapter.update_weights(weights, global_steps=12, preserve_kv_cache=True))
            except RuntimeError as exc:
                return str(exc)
            return "UNEXPECTED_SUCCESS"

        self.transport.side_effect = reject
        with (
            patch.object(dist, "gather_object", side_effect=gather),
            patch.object(dist, "broadcast_object_list", side_effect=broadcast),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            outcomes = [future.result(timeout=10) for future in [pool.submit(participant, rank) for rank in (0, 1)]]
        self.assertEqual(gather_counts, [expected_gathers, expected_gathers])
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertIn("failed during preserved weight update", outcomes[0])
        self.assertNotIn("/private/native/detail", outcomes[0])
        self.assert_no_publication_or_cleanup()
        if failure != "end_weight_update":
            self.assertNotIn("end_weight_update", [event[0] for event in self.events])

    def test_direct_native_api_does_not_require_session_endpoints(self):
        with patch.object(rollout, "_preserved_weight_update_uses_session", return_value=False):
            self.run_update(global_steps=12, preserve_kv_cache=True)
        self.assertEqual(
            [event[0] for event in self.events],
            ["update_weights_from_tensor", "update_weights_from_tensor", "publish"],
        )
        for _, payload in self.events[:-1]:
            self.assertIs(payload["flush_cache"], False)
            self.assertIs(payload["abort_all_requests"], False)
        self.engine.flush_cache.assert_not_awaited()


class TestNativePreservationCapabilities(unittest.TestCase):
    @staticmethod
    def schema(begin=False, end=False, missing=None):
        pause = type("Pause", (), {"__annotations__": {"mode": Literal["abort", "retract", "in_place"]}})
        fields = {"flush_cache": bool, "abort_all_requests": bool}
        if missing is not None:
            del fields[missing]
        tensor = type("TensorUpdate", (), {"__annotations__": fields})
        api = SimpleNamespace(PauseGenerationReqInput=pause, UpdateWeightsFromTensorReqInput=tensor)
        if begin:
            api.BeginWeightUpdateReqInput = object
        if end:
            api.EndWeightUpdateReqInput = object
        return api

    def test_released_direct_api(self):
        with patch.object(managers, "io_struct", self.schema()):
            self.assertFalse(rollout._preserved_weight_update_uses_session())

    def test_paired_session_api(self):
        with patch.object(managers, "io_struct", self.schema(begin=True, end=True)):
            self.assertTrue(rollout._preserved_weight_update_uses_session())

    def test_incomplete_session_is_rejected(self):
        for begin, end in [(True, False), (False, True)]:
            with self.subTest(begin=begin, end=end):
                with patch.object(managers, "io_struct", self.schema(begin=begin, end=end)):
                    with self.assertRaisesRegex(RuntimeError, "session schema is incomplete"):
                        rollout._preserved_weight_update_uses_session()

    def test_missing_preservation_flags_are_rejected(self):
        for field in ["flush_cache", "abort_all_requests"]:
            with self.subTest(field=field):
                with patch.object(managers, "io_struct", self.schema(missing=field)):
                    with self.assertRaisesRegex(RuntimeError, "lacks preservation flags"):
                        rollout._preserved_weight_update_uses_session()

    def test_retract_only_api_is_rejected(self):
        api = self.schema()
        api.PauseGenerationReqInput.__annotations__["mode"] = Literal["abort", "retract"]
        with patch.object(managers, "io_struct", api):
            with self.assertRaisesRegex(RuntimeError, "in_place pause"):
                rollout._preserved_weight_update_uses_session()


if __name__ == "__main__":
    unittest.main()
