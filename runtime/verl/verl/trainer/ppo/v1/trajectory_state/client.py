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

"""Picklable async facade; callbacks retain an ActorHandle, never file locks."""

from __future__ import annotations


class RayTrajectoryCoordinatorClient:
    def __init__(self, actor):
        self.actor = actor

    async def register_producer(self, producer_id, incarnation_id):
        return await self.actor.register_producer.remote(producer_id, incarnation_id)

    async def get_current_producer_generation(self, producer_id):
        return await self.actor.get_current_producer_generation.remote(producer_id)

    async def open_trajectory(self, trajectory_id, owner_id, lease_seconds):
        return await self.actor.open_trajectory.remote(trajectory_id, owner_id, lease_seconds)

    async def register_running_trajectory(self, trajectory_id):
        return await self.actor.register_running_trajectory.remote(trajectory_id)

    async def register_owned_running_trajectory(self, owner_lease):
        return await self.actor.register_owned_running_trajectory.remote(owner_lease)

    async def request_trajectory_preemption(self, trajectory_id):
        return await self.actor.request_trajectory_preemption.remote(trajectory_id)

    async def request_owned_trajectory_preemption(self, owner_lease):
        return await self.actor.request_owned_trajectory_preemption.remote(owner_lease)

    async def get_trajectory_preemption(self, trajectory_id):
        return await self.actor.get_trajectory_preemption.remote(trajectory_id)

    async def mark_trajectory_safe_point(self, trajectory_id):
        return await self.actor.mark_trajectory_safe_point.remote(trajectory_id)

    async def mark_owned_trajectory_safe_point(self, owner_lease):
        return await self.actor.mark_owned_trajectory_safe_point.remote(owner_lease)

    async def check_trajectory_owner(self, owner_lease):
        return await self.actor.check_trajectory_owner.remote(owner_lease)

    async def renew_trajectory_owner(self, owner_lease, lease_seconds):
        return await self.actor.renew_trajectory_owner.remote(owner_lease, lease_seconds)

    async def commit_trajectory_snapshot(self, owner_lease, snapshot, ttl_seconds):
        return await self.actor.commit_trajectory_snapshot.remote(owner_lease, snapshot, ttl_seconds)

    async def mark_trajectory_snapshot_committed(self, trajectory_id, snapshot_ref):
        return await self.actor.mark_trajectory_snapshot_committed.remote(trajectory_id, snapshot_ref)

    async def put_handoff_member_record(self, payload):
        return await self.actor.put_handoff_member_record.remote(payload)

    async def get_handoff_member_record(self, reference):
        return await self.actor.get_handoff_member_record.remote(reference)

    async def prepare_handoff_manifest(self, manifest):
        return await self.actor.prepare_handoff_manifest.remote(manifest)

    async def commit_handoff_manifest(self, handoff_id):
        return await self.actor.commit_handoff_manifest.remote(handoff_id)

    async def get_handoff_manifest(self, handoff_id):
        return await self.actor.get_handoff_manifest.remote(handoff_id)

    async def get_handoff_status(self, handoff_id):
        return await self.actor.get_handoff_status.remote(handoff_id)

    async def list_recoverable_handoffs(self):
        return await self.actor.list_recoverable_handoffs.remote()

    async def release_trajectory_owner(self, owner_lease):
        return await self.actor.release_trajectory_owner.remote(owner_lease)

    async def finish_trajectory(self, owner_lease):
        return await self.actor.finish_trajectory.remote(owner_lease)

    async def discard_trajectory_preemption(self, trajectory_id):
        return await self.actor.discard_trajectory_preemption.remote(trajectory_id)

    async def mark_trajectory_aborted(self, trajectory_id):
        return await self.actor.mark_trajectory_aborted.remote(trajectory_id)

    async def abort_owned_trajectory(self, owner_lease):
        return await self.actor.abort_owned_trajectory.remote(owner_lease)

    async def release_handoff_source(self, handoff_id):
        return await self.actor.release_handoff_source.remote(handoff_id)

    async def record_handoff_source_release(self, handoff_id):
        return await self.actor.record_handoff_source_release.remote(handoff_id)

    async def mark_trajectory_handed_off(self, trajectory_id):
        return await self.actor.mark_trajectory_handed_off.remote(trajectory_id)

    async def begin_handoff_restore(self, handoff_id):
        return await self.actor.begin_handoff_restore.remote(handoff_id)

    async def acquire_handoff_trajectory(self, handoff_id, trajectory_id, owner_id, lease_seconds):
        return await self.actor.acquire_handoff_trajectory.remote(handoff_id, trajectory_id, owner_id, lease_seconds)

    async def acquire_handoff_pending(self, handoff_id, trajectory_id, owner_id, lease_seconds):
        return await self.actor.acquire_handoff_pending.remote(handoff_id, trajectory_id, owner_id, lease_seconds)

    async def load_trajectory_snapshot(self, owner_lease):
        return await self.actor.load_trajectory_snapshot.remote(owner_lease)

    async def record_handoff_member_result(self, handoff_id, member_index, owner_lease, payload, metrics):
        return await self.actor.record_handoff_member_result.remote(
            handoff_id, member_index, owner_lease, payload, metrics
        )

    async def get_handoff_restored_members(self, handoff_id):
        return await self.actor.get_handoff_restored_members.remote(handoff_id)

    async def prepare_publication(self, handoff_id, stable_binding):
        return await self.actor.prepare_publication.remote(handoff_id, stable_binding)

    async def mark_publication_started(self, handoff_id):
        return await self.actor.mark_publication_started.remote(handoff_id)

    async def confirm_publication(self, handoff_id, receipt):
        return await self.actor.confirm_publication.remote(handoff_id, receipt)

    async def reconcile_publication(self, handoff_id, receipt):
        return await self.actor.reconcile_publication.remote(handoff_id, receipt)

    async def get_publication(self, handoff_id):
        return await self.actor.get_publication.remote(handoff_id)

    async def finish_handoff_manifest(self, handoff_id, publication_id):
        return await self.actor.finish_handoff_manifest.remote(handoff_id, publication_id)

    async def queue_completed_group_cleanup(self, uid, owner_id, source_generation,
                                            input_record_refs, output_record_refs):
        return await self.actor.queue_completed_group_cleanup.remote(
            uid, owner_id, source_generation, input_record_refs, output_record_refs
        )

    async def collect_garbage(self):
        return await self.actor.collect_garbage.remote()

    async def summary(self):
        return await self.actor.summary.remote()

    async def export_checkpoint(self, destination, checkpoint_step):
        return await self.actor.export_checkpoint.remote(destination, checkpoint_step)

    async def get_trajectory_preemption_metrics(self):
        return await self.actor.get_trajectory_preemption_metrics.remote()

    async def get_handoff_metrics(self):
        return await self.actor.get_handoff_metrics.remote()

    async def close(self):
        return await self.actor.close.remote()

    async def begin_policy_publication(self, identity, model_configuration_fingerprint, publisher_id, engine_urls):
        return await self.actor.begin_policy_publication.remote(
            identity, model_configuration_fingerprint, publisher_id, engine_urls)

    async def mark_policy_update_started(self, ticket):
        return await self.actor.mark_policy_update_started.remote(ticket)

    async def confirm_policy_publication(self, ticket, observations):
        return await self.actor.confirm_policy_publication.remote(ticket, observations)

    async def fail_policy_publication(self, ticket, reason):
        return await self.actor.fail_policy_publication.remote(ticket, reason)

    async def get_published_policy(self, version):
        return await self.actor.get_published_policy.remote(version)

    async def get_active_published_policy(self):
        return await self.actor.get_active_published_policy.remote()

    async def authorize_same_version_engines(self, version, engine_urls, observations):
        return await self.actor.authorize_same_version_engines.remote(version, engine_urls, observations)

    async def get_policy_coordination_state(self):
        return await self.actor.get_policy_coordination_state.remote()

    async def begin_handoff_policy_barrier(self, group_id, source_leases, timeout_seconds):
        return await self.actor.begin_handoff_policy_barrier.remote(group_id, source_leases, timeout_seconds)

    async def check_handoff_policy_barrier(self, barrier):
        return await self.actor.check_handoff_policy_barrier.remote(barrier)

    async def release_handoff_policy_barrier(self, barrier, handoff_id=None):
        return await self.actor.release_handoff_policy_barrier.remote(barrier, handoff_id)
