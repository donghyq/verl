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

"""Check only this actor's descendants without signalling unrelated processes."""

import time

import psutil


def wait_owned_children(timeout=30):
    """Wait for live descendants; zombies retain exit status, not GPU resources."""
    deadline = time.monotonic() + timeout
    observed_zombies = set()
    while True:
        live = []
        states = []
        for child in psutil.Process().children(recursive=True):
            try:
                status = child.status()
                states.append({"pid": child.pid, "status": status})
                if status == psutil.STATUS_ZOMBIE:
                    observed_zombies.add(child.pid)
                elif child.is_running():
                    live.append(child.pid)
            except psutil.NoSuchProcess:
                continue
        if not live:
            return {"owned_children_remaining": [], "observed_zombies": sorted(observed_zombies), "states": states}
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Owned engine live children remain: {states}")
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
