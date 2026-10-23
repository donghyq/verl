# Copyright 2026 Individual Contributor: donghyq
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

"""Actual child-process tests; no CUDA, Ray, or SGLang imports."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[3] / "special_distributed" / "sglang_preserve_process.py"


@pytest.mark.parametrize("case", ["zombie", "live", "delayed", "reaped"])
def test_actual_owned_child_lifecycle(case):
    code = r"""
import importlib.util,json,subprocess,sys,time,psutil
spec=importlib.util.spec_from_file_location('owned',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
case=sys.argv[2]
delay='30' if case=='live' else '0.2' if case=='delayed' else '0'
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep('+delay+')'])
try:
 if case=='live':
  try:m.wait_owned_children(timeout=0)
  except RuntimeError:print(json.dumps({'case':case,'live_rejected':True}))
  else:raise AssertionError('Live child incorrectly accepted')
 else:
  if case=='reaped':p.wait(timeout=2)
  elif case=='zombie':
   deadline=time.monotonic()+2
   while psutil.Process(p.pid).status()!=psutil.STATUS_ZOMBIE:
    assert time.monotonic()<deadline;time.sleep(.01)
   # The old check rejects a real, already-exited child.
   assert psutil.Process(p.pid).is_running()
  result=m.wait_owned_children(timeout=2)
  assert not result['owned_children_remaining']
  if case=='zombie':assert p.pid in result['observed_zombies']
  print(json.dumps({'case':case,'result':result}))
finally:
 if p.poll() is None:p.terminate()
 p.wait(timeout=2)
"""
    result = subprocess.run([sys.executable, "-c", code, str(HELPER), case], capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["case"] == case
