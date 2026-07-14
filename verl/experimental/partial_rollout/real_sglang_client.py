"""HTTP client wrapper for a real SGLang server.

Drop-in replacement for :class:`FakeSGLangServer` that calls a real SGLang
server's HTTP API. Used by :class:`SGLangPartialRolloutCoordinator` to drive
the full pause → save → resume → complete loop against a live server.

Usage:
    from verl.experimental.partial_rollout.real_sglang_client import RealSGLangServer
    from verl.experimental.partial_rollout.sglang_integration import SGLangPartialRolloutCoordinator

    server = RealSGLangServer("http://localhost:30000")
    coord = SGLangPartialRolloutCoordinator(server=server, model_weight_version="v0")
    # ... pause/resume/complete against the real server
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from verl.experimental.partial_rollout.rollout_state import new_kv_handle

logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    requests = None  # type: ignore


class RealSGLangServer:
    """HTTP client for a real SGLang server, matching FakeSGLangServer's interface.

    Each method maps to a real HTTP endpoint:
    - pause_generation → POST /pause_generation
    - release_memory_occupation → POST /release_memory_occupation
    - resume_memory_occupation → POST /resume_memory_occupation
    - pause_request → POST /pause_request
    - resume_request → POST /resume_request
    - generate → POST /v1/chat/completions
    """

    def __init__(self, base_url: str, timeout: int = 60) -> None:
        if requests is None:
            raise ImportError("requests library is required: pip install requests")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.weights_loaded: bool = True
        self.resident_kv: dict[str, str] = {}
        self.released: set[str] = set()
        self.paused: set[str] = set()
        self.call_log: list[str] = []
        self._weight_version: str = "v0"

    def _post(self, path: str, payload: dict) -> dict:
        resp = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> dict:
        resp = requests.get(f"{self.base_url}{path}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    # server-side API surface (matches FakeSGLangServer)
    # ------------------------------------------------------------------ #

    def pause_generation(self, request_id: str, mode: str = "preserve_kv") -> None:
        self.call_log.append(f"pause_generation({request_id}, mode={mode})")
        self.paused.add(request_id)
        self._post("/pause_generation", {"mode": mode})

    def release_memory_occupation(self, request_id: str, tags: list[str]) -> None:
        self.call_log.append(f"release_memory_occupation({request_id}, tags={tags})")
        if "kv_cache" in tags:
            self.resident_kv.pop(request_id, None)
            self.released.add(request_id)
        if "weights" in tags:
            self.weights_loaded = False
        self._post("/release_memory_occupation", {"tags": tags})

    def resume_memory_occupation(self, request_id: str, tags: list[str]) -> None:
        self.call_log.append(f"resume_memory_occupation({request_id}, tags={tags})")
        if "weights" in tags:
            self.weights_loaded = True
        self._post("/resume_memory_occupation", {"tags": tags})

    def allocate_kv(self, request_id: str) -> str:
        handle = new_kv_handle()
        self.resident_kv[request_id] = handle
        self.call_log.append(f"allocate_kv({request_id})")
        return handle

    def is_kv_resident(self, request_id: str) -> bool:
        return request_id in self.resident_kv

    @property
    def last_call(self) -> str:
        return self.call_log[-1] if self.call_log else ""

    # ------------------------------------------------------------------ #
    # additional real-server operations
    # ------------------------------------------------------------------ #

    def continue_generation(self, torch_empty_cache: bool = True) -> None:
        self.call_log.append("continue_generation()")
        self.paused.clear()
        self._post("/continue_generation", {"torch_empty_cache": torch_empty_cache})

    def pause_request(self, request_id: str, pause_all: bool = False) -> None:
        self.call_log.append(f"pause_request({request_id}, pause_all={pause_all})")
        self.paused.add(request_id)
        self._post("/pause_request", {"rid": request_id, "pause_all": pause_all})

    def resume_request(self, request_id: str = "", resume_all: bool = True) -> None:
        self.call_log.append(f"resume_request({request_id}, resume_all={resume_all})")
        self.paused.discard(request_id)
        self._post("/resume_request", {"rid": request_id, "resume_all": resume_all})

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        request_id: Optional[str] = None,
    ) -> dict:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if request_id:
            payload["rid"] = request_id
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
        request_id: Optional[str] = None,
    ):
        """Stream generation, yields (first_token_time, total_time, token_count)."""
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if request_id:
            payload["rid"] = request_id

        start = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()

        first_token_time = None
        token_count = 0
        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    import json
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            if first_token_time is None:
                                first_token_time = time.perf_counter() - start
                            token_count += 1
                    except Exception:
                        pass
        total_time = time.perf_counter() - start
        return first_token_time or 0.0, total_time, token_count

    def get_server_info(self) -> dict:
        return self._get("/get_server_info")

    def health_check(self) -> bool:
        try:
            self._get("/health")
            return True
        except Exception:
            return False

    def update_weight_version(self, version: str) -> None:
        self._weight_version = version
        self.weights_loaded = False
        self.call_log.append(f"update_weight_version({version})")
