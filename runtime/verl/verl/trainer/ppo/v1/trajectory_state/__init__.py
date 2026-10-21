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

"""Opt-in state handoff; stock V1 queue and training remain the default."""

from .coordinator import DurableTrajectoryCoordinator, PublicationBlocked
from .client import RayTrajectoryCoordinatorClient
from .handoff import HandoffManifest, decode_handoff_member_record, encode_handoff_member_record
from .identity import CanonicalPolicyKVIdentity
from .snapshot import TrajectoryOwnerLease, TrajectorySnapshot
from .storage import CoordinationStorageError
from .policy_publication import (
    EnginePolicyObservation, HandoffPublicationBarrier, PolicyPublicationBlocked,
    PolicyPublicationDriver, PolicyPublicationTicket, PublishedPolicyRecord,
)

__all__ = [
    "CanonicalPolicyKVIdentity",
    "CoordinationStorageError",
    "DurableTrajectoryCoordinator",
    "HandoffManifest",
    "PublicationBlocked",
    "RayTrajectoryCoordinatorClient",
    "EnginePolicyObservation",
    "HandoffPublicationBarrier",
    "PolicyPublicationBlocked",
    "PolicyPublicationDriver",
    "PolicyPublicationTicket",
    "PublishedPolicyRecord",
    "TrajectoryOwnerLease",
    "TrajectorySnapshot",
    "create_ray_trajectory_coordinator_actor",
    "decode_handoff_member_record",
    "encode_handoff_member_record",
]


def create_ray_trajectory_coordinator_actor(*, max_restarts: int = 0):
    """Return the serialized Ray actor class, without importing Ray for codecs."""
    if type(max_restarts) is not int or max_restarts < 0:
        raise ValueError("max_restarts must be a non-negative integer")
    import ray

    return ray.remote(num_cpus=1, max_concurrency=1, max_restarts=max_restarts)(DurableTrajectoryCoordinator)
