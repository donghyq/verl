# Copyright 2026 Individual Contributor: Hanyudong
# Licensed under the Apache License, Version 2.0.
"""Existing bounded Conductor client, copied without class-body changes."""
import http.client
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

class ConductorHTTPClient:
    """Bounded native HTTP/msgpack query; no cached view survives an error."""

    def __init__(self, endpoint: str, timeout_ms: float = 50):
        if not endpoint.startswith(("http://", "https://")) or not 0 < timeout_ms <= 1000:
            raise ValueError("Conductor needs an HTTP endpoint and timeout_ms in (0, 1000]")
        self.endpoint = endpoint.rstrip("/")
        self._url = urlsplit(self.endpoint)
        if not self._url.hostname or self._url.username or self._url.query or self._url.fragment:
            raise ValueError("Conductor endpoint needs a host and no credentials, query or fragment")
        self.timeout = timeout_ms / 1000
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="conductor-query")
        self._pending = None
        self._socket_lock = threading.Lock()
        self._socket = None
        self._cancelled = threading.Event()
        self._closed = False

    def query(self, payload: dict) -> dict:
        # The Ray actor calls this serially. One pending operation also bounds
        # resolver work, which Python cannot interrupt. For established sockets,
        # shutdown interrupts even a peer trickling headers or response bytes.
        if self._closed:
            raise RuntimeError("Conductor client is closed")
        if self._pending is not None and not self._pending.done():
            raise TimeoutError("Previous Conductor query is still running")
        body = self._encode_request(payload)
        self._cancelled = threading.Event()
        self._pending = self._executor.submit(self._post, body)
        try:
            return self._pending.result(timeout=self.timeout)
        except TimeoutError:
            self._cancel_io()
            raise

    def _post(self, body: bytes) -> dict:
        connection_type = http.client.HTTPSConnection if self._url.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(self._url.hostname, self._url.port, timeout=self.timeout)
        active_socket = None
        try:
            connection.connect()
            with self._socket_lock:
                if self._cancelled.is_set():
                    raise TimeoutError("Conductor query expired before connection completed")
                active_socket = self._socket = connection.sock
            path, wire_body, headers = self._wire_request(body)
            connection.request("POST", path, body=wire_body, headers=headers)
            with connection.getresponse() as response:
                if response.status != 200:
                    raise ValueError(f"Conductor returned HTTP {response.status}")
                data = response.read(2 * 1024 * 1024 + 1)
        finally:
            with self._socket_lock:
                if self._socket is active_socket:
                    self._socket = None
            connection.close()
        if len(data) > 2 * 1024 * 1024:
            raise ValueError("Conductor response exceeds 2 MiB")
        return self._decode_response(data)

    def _encode_request(self, payload: dict):
        import msgpack
        return msgpack.packb(payload, use_bin_type=True)

    def _wire_request(self, body: bytes):
        return (self._url.path.rstrip("/") + "/query", body,
                {"Content-Type": "application/msgpack"})

    def _decode_response(self, data: bytes):
        import msgpack
        result = msgpack.unpackb(data, raw=False, strict_map_key=False)
        if not isinstance(result, dict) or not isinstance(result.get("instances"), dict):
            raise ValueError("Conductor returned no instance map")
        return result["instances"]

    def _cancel_io(self) -> None:
        with self._socket_lock:
            self._cancelled.set()
            if self._socket is not None:
                try:
                    self._socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def close(self) -> None:
        self._closed = True
        self._cancel_io()
        self._executor.shutdown(wait=False, cancel_futures=True)
