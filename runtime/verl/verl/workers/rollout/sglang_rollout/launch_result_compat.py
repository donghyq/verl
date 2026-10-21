# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Normalize the two SGLang subprocess-launch return contracts."""

from typing import Any, NamedTuple


class LaunchResult(NamedTuple):
    tokenizer_manager: Any
    template_manager: Any
    scheduler_info: dict | None
    port_args: Any
    watchdog_thread: Any


def unpack_launch_result(result: tuple | list, *, node_rank: int) -> LaunchResult:
    """Require rank zero's actual scheduler dictionary before publishing HTTP.

    Older SGLang returns (tokenizer, template, scheduler_info). Newer SGLang
    inserts PortArgs before either a dictionary or SchedulerInitResult.
    Nonzero nodes do not own the tokenizer or an HTTP server and may return None.
    """
    if not isinstance(result, (tuple, list)) or len(result) < 3:
        raise TypeError("Unsupported SGLang subprocess launch result")

    tokenizer, template, third = result[:3]
    port_args = watchdog = None
    if isinstance(third, dict) or (third is None and len(result) == 3):
        scheduler_info = third
    elif len(result) >= 4 and isinstance(result[3], dict):
        port_args, scheduler_info = third, result[3]
        if len(result) >= 5:
            watchdog = result[4]
    elif len(result) >= 4 and hasattr(result[3], "scheduler_infos"):
        port_args = third
        scheduler_infos = result[3].scheduler_infos
        if scheduler_infos is None and node_rank > 0:
            scheduler_info = None
        elif isinstance(scheduler_infos, (tuple, list)) and scheduler_infos:
            scheduler_info = scheduler_infos[0]
        else:
            raise TypeError("SGLang SchedulerInitResult has no scheduler dictionary")
        if len(result) >= 5:
            watchdog = result[4]
    elif len(result) >= 4 and result[3] is None and node_rank > 0:
        port_args, scheduler_info = third, None
        if len(result) >= 5:
            watchdog = result[4]
    else:
        raise TypeError("Unknown SGLang subprocess launch return layout")

    if not isinstance(scheduler_info, dict) and not (scheduler_info is None and node_rank > 0):
        raise TypeError("SGLang HTTP rank requires scheduler_info to be a dictionary")
    return LaunchResult(tokenizer, template, scheduler_info, port_args, watchdog)
