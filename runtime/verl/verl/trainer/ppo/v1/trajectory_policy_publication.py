# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Minimal root-entry adapters for continuous policy publication.

This module only connects existing checkpoint/Gateway lifecycle hooks to the
existing trajectory coordinator publication API. It does not add a second
version table, route generation, or change weight transfer semantics.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .trajectory_state import CanonicalPolicyKVIdentity, PolicyPublicationDriver


def _mapping_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict")
    return dict(value)


def canonical_identity_from_runtime(*, policy_version: int | str, runtime_identity: dict[str, Any]) -> dict[str, str]:
    """Build the published identity from actual runtime/model configuration fields.

    The caller must pass the actual runtime-derived fingerprints for model
    weights lineage, tokenizer/template and KV schema. This function only pins
    the current committed policy version onto that already-validated runtime
    identity.
    """

    identity = _mapping_dict(runtime_identity, "runtime_identity")
    canonical = CanonicalPolicyKVIdentity(
        policy_version=str(policy_version),
        model_weights_fingerprint=identity["model_weights_fingerprint"],
        tokenizer_template_fingerprint=identity["tokenizer_template_fingerprint"],
        kv_cache_schema_fingerprint=identity["kv_cache_schema_fingerprint"],
    )
    canonical.validate()
    return asdict(canonical)


class TrajectoryPolicyPublicationGatewayHooks:
    """Checkpoint/Gateway-facing adapter around PolicyPublicationDriver."""

    def __init__(
        self,
        publication_client,
        *,
        run_id: str,
        runtime_policy_identity: dict[str, Any],
        model_configuration_fingerprint: str,
        publisher_id: str,
        engine_urls: list[str],
        engine_observer,
    ) -> None:
        self.driver = PolicyPublicationDriver(publication_client)
        self.run_id = run_id
        self.runtime_policy_identity = _mapping_dict(runtime_policy_identity, "runtime_policy_identity")
        self.model_configuration_fingerprint = model_configuration_fingerprint
        self.publisher_id = publisher_id
        self.engine_urls = list(engine_urls)
        self.engine_observer = engine_observer

    async def prepare_policy_publication(self, global_steps: int):
        identity = canonical_identity_from_runtime(
            policy_version=global_steps,
            runtime_identity=self.runtime_policy_identity,
        )
        return await self.driver.prepare(
            identity=identity,
            model_configuration_fingerprint=self.model_configuration_fingerprint,
            publisher_id=self.publisher_id,
            engine_urls=self.engine_urls,
        )

    async def mark_policy_update_started(self, ticket) -> None:
        await self.driver.mark_update_started(ticket)

    async def confirm_policy_publication_before_admission(self, ticket):
        return await self.driver.confirm_before_admission(ticket, self.engine_observer)

    async def fail_policy_publication(self, ticket, reason: str) -> None:
        await self.driver.fail(ticket, reason)
