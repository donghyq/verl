# Copyright 2026 Individual Contributor: donghyq
# Licensed under the Apache License, Version 2.0.
"""Synthetic history invariants; no CUDA or native scheduler import."""

import copy
import importlib.util
from pathlib import Path

import pytest

p = Path(__file__).resolve().parents[3] / "special_distributed" / "sglang_preserve_history.py"
spec = importlib.util.spec_from_file_location("native_history", p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
classify = module.classify_native_history


def rows():
    return [
        {"output_interval": [0, 1], "positions": [0, 1, 2], "forward_id": 1},
        {"output_interval": [1, 2], "positions": [3], "forward_id": 2},
        {"output_interval": [2, 3], "positions": [4], "forward_id": 3},
        {"output_interval": [3, 3], "positions": [5], "forward_id": 4},
    ]


@pytest.mark.parametrize("tail", [False, True])
def test_exact_history_with_optional_native_overlap_tail(tail):
    observed = rows() if tail else rows()[:-1]
    productive, discarded = classify(observed, 3, 3, True, True)
    assert len(productive) == 3 and len(discarded) == int(tail)
    assert sum(len(r["positions"]) for r in productive + discarded) == 5 + int(tail)


@pytest.mark.parametrize(
    "corruption",
    [
        "repeat",
        "gap",
        "reorder",
        "missing_output",
        "repeat_interval",
        "wrong_tail",
        "tail_interval",
        "tail_order",
        "intermediate",
        "multiple",
        "regress",
    ],
)
def test_reject_unexplained_history_or_output(corruption):
    observed = copy.deepcopy(rows())
    if corruption == "repeat":
        observed[1]["positions"] = [2]
    elif corruption == "gap":
        observed[1]["positions"] = [6]
    elif corruption == "reorder":
        observed[0], observed[1] = observed[1], observed[0]
    elif corruption == "missing_output":
        observed.pop(1)
    elif corruption == "repeat_interval":
        observed[1]["output_interval"] = [0, 1]
    elif corruption == "wrong_tail":
        observed[-1]["positions"] = [4]
    elif corruption == "tail_interval":
        observed[-1]["output_interval"] = [2, 2]
    elif corruption == "tail_order":
        observed[-1]["forward_id"] = 3
    elif corruption == "intermediate":
        observed[-1], observed[-2] = observed[-2], observed[-1]
    elif corruption == "multiple":
        observed.append(copy.deepcopy(observed[-1]))
    elif corruption == "regress":
        observed[-1]["output_interval"] = [3, 2]
    with pytest.raises(AssertionError):
        classify(observed, 3, 3, True, True)


@pytest.mark.parametrize("overlap,finished", [(False, True), (True, False)])
def test_reject_tail_outside_finished_overlap(overlap, finished):
    with pytest.raises(AssertionError):
        classify(rows(), 3, 3, overlap, finished)
