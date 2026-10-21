# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Keep control operations alive across synchronous manager API calls."""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
import weakref


class TrajectoryControlDispatcher:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="trajectory-control", daemon=True)
        self.thread.start()
        self._ready.wait()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    def submit(self, coroutine):
        if self.loop.is_closed():
            coroutine.close()
            raise RuntimeError("trajectory control dispatcher is closed")
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def close(self):
        """Stop an idle dispatcher; never cancel dispatched operations."""
        if self.loop.is_closed():
            return

        def stop_if_idle():
            if not asyncio.all_tasks(self.loop):
                self.loop.stop()

        self.loop.call_soon_threadsafe(stop_if_idle)


_dispatcher_lock = threading.Lock()


def _dispatcher(manager):
    with _dispatcher_lock:
        existing = getattr(manager, "_trajectory_control_dispatcher", None)
        if existing is None:
            existing = TrajectoryControlDispatcher()
            manager._trajectory_control_dispatcher = existing
            weakref.finalize(manager, existing.close)
        return existing


def trajectory_control_api(function):
    """Preserve sync/await syntax while using one persistent control loop."""

    @functools.wraps(function)
    def call(manager, *args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        existing = getattr(manager, "_trajectory_control_dispatcher", None)
        if existing is not None and loop is existing.loop:
            return function(manager, *args, **kwargs)

        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        caller_is_async = caller is not None and bool(caller.f_code.co_flags & inspect.CO_COROUTINE)
        del frame, caller
        if loop is not None and caller_is_async:

            async def awaited():
                dispatcher = _dispatcher(manager)
                future = dispatcher.submit(function(manager, *args, **kwargs))
                return await asyncio.wrap_future(future)

            return awaited()
        dispatcher = _dispatcher(manager)
        return dispatcher.submit(function(manager, *args, **kwargs)).result()

    return call
