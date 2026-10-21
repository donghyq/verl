# Copyright 2026 Project B contributors
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

"""The existing producer generation rules, without an Artifact queue."""
from __future__ import annotations


class ProducerGenerationRegistry:
    def __init__(self, incarnations: dict[str, dict[str, int]]):
        self._producer_incarnations = incarnations

    def register_producer(self, producer_id: str, incarnation_id: str) -> int:
        if not isinstance(producer_id, str) or not producer_id:
            raise ValueError("producer_id must be a non-empty string")
        if not isinstance(incarnation_id, str) or not incarnation_id:
            raise ValueError("incarnation_id must be a non-empty string")
        incarnations = self._producer_incarnations.setdefault(producer_id, {})
        existing_generation = incarnations.get(incarnation_id)
        if existing_generation is not None:
            current_generation = max(incarnations.values())
            if existing_generation != current_generation:
                raise ValueError(
                    f"retired producer incarnation is not current: {producer_id}/{incarnation_id}"
                )
            return existing_generation
        generation = 0 if not incarnations else max(incarnations.values()) + 1
        incarnations[incarnation_id] = generation
        return generation

    def get_current_producer_generation(self, producer_id: str) -> int | None:
        incarnations = self._producer_incarnations.get(producer_id)
        return None if not incarnations else max(incarnations.values())
