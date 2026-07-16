"""Keep the dependency-free partial-rollout tests isolated from VeRL imports."""

from __future__ import annotations

import sys
import types
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]

for _name, _path in (
    ("verl", _REPO_ROOT / "verl"),
    ("verl.experimental", _REPO_ROOT / "verl" / "experimental"),
):
    if _name not in sys.modules:
        _module = types.ModuleType(_name)
        _module.__path__ = [str(_path)]
        _module.__package__ = _name
        sys.modules[_name] = _module
