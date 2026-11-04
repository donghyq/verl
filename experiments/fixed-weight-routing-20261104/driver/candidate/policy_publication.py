# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Bind original checkpoint publications to observed policy identities.

These are small coordination records, not weight checkpoints or a scheduler.
The original checkpoint implementation owns parameter transfer. A version is
usable only after that call completes and every configured engine reports it.
The model fingerprint identifies the configured model lineage; it must not be
described as a digest of updated tensor bytes.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable

from .identity import CanonicalPolicyKVIdentity, _decode_policy_identity


class PolicyPublicationBlocked(RuntimeError):
    """Publication or a bounded handoff cut has an unresolved outcome."""


def policy_version(value: Any) -> str:
    if type(value) is int and value >= 0:
        return str(value)
    if isinstance(value, str) and value.isascii() and value.isdecimal() and value == str(int(value)):
        return value
    raise ValueError("policy version must be the original non-negative committed training step")


def nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class EnginePolicyObservation:
    engine_url: str
    weight_version: str
    source: str = "model_info"

    def __post_init__(self):
        nonempty(self.engine_url, "engine_url")
        policy_version(self.weight_version)
        if self.source != "model_info":
            raise ValueError("only actual engine model_info version observations are accepted")

    @classmethod
    def decode(cls, value: dict) -> "EnginePolicyObservation":
        if not isinstance(value, dict) or set(value) != {"engine_url", "weight_version", "source"}:
            raise ValueError("invalid engine version observation")
        return cls(**value)


@dataclass(frozen=True)
class PublishedPolicyRecord:
    run_id: str
    identity: CanonicalPolicyKVIdentity
    model_configuration_fingerprint: str
    publisher_id: str
    engine_observations: tuple[EnginePolicyObservation, ...]
    committed_at_unix: float

    def __post_init__(self):
        for name in ("run_id", "model_configuration_fingerprint", "publisher_id"):
            nonempty(getattr(self, name), name)
        self.identity.validate()
        policy_version(self.identity.policy_version)
        if (not self.engine_observations
                or len({row.engine_url for row in self.engine_observations}) != len(self.engine_observations)
                or any(row.weight_version != self.identity.policy_version for row in self.engine_observations)):
            raise ValueError("published policy requires consistent actual observations for all distinct engines")
        if (isinstance(self.committed_at_unix, bool)
                or not isinstance(self.committed_at_unix, (int, float))
                or not math.isfinite(self.committed_at_unix)):
            raise ValueError("invalid policy publication time")

    def encode(self) -> dict:
        return asdict(self)

    @classmethod
    def decode(cls, value: dict) -> "PublishedPolicyRecord":
        expected = {"run_id", "identity", "model_configuration_fingerprint", "publisher_id",
                    "engine_observations", "committed_at_unix"}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("invalid published policy record")
        return cls(value["run_id"], _decode_policy_identity(value["identity"]),
                   value["model_configuration_fingerprint"], value["publisher_id"],
                   tuple(EnginePolicyObservation.decode(row) for row in value["engine_observations"]),
                   value["committed_at_unix"])


@dataclass(frozen=True)
class PolicyPublicationTicket:
    run_id: str
    operation_id: str
    policy_version: str


@dataclass(frozen=True)
class HandoffPublicationBarrier:
    run_id: str
    operation_id: str
    group_id: str
    identity: CanonicalPolicyKVIdentity


def empty_policy_state() -> dict:
    return {"records": {}, "active_version": None, "publication": None, "barrier": None}


def validate_policy_state(state: dict, run_id: str) -> None:
    if not isinstance(state, dict) or set(state) != {"records", "active_version", "publication", "barrier"}:
        raise ValueError("invalid persisted policy publication state")
    if not isinstance(state["records"], dict):
        raise ValueError("invalid policy publication records")
    for version, raw in state["records"].items():
        record = PublishedPolicyRecord.decode(raw)
        if record.run_id != run_id or record.identity.policy_version != policy_version(version):
            raise ValueError("policy publication record scope differs")
    active = state["active_version"]
    if active is not None and (active not in state["records"]
                               or int(active) != max(map(int, state["records"]))):
        raise ValueError("active policy does not match committed publication history")
    if active is None and state["records"]:
        raise ValueError("committed policy history has no active identity")
    if state["publication"] is not None and state["barrier"] is not None:
        raise ValueError("parameter publication and handoff cut cannot overlap")
    publication = state["publication"]
    if publication is not None:
        expected = {"operation_id", "version", "identity", "model_configuration_fingerprint", "publisher_id",
                    "engine_urls", "state", "failure"}
        if (not isinstance(publication, dict) or set(publication) != expected
                or publication["state"] not in {"PREPARED", "UPDATING", "BLOCKED"}
                or _decode_policy_identity(publication["identity"]).policy_version != publication["version"]
                or not publication["engine_urls"]
                or len(set(publication["engine_urls"])) != len(publication["engine_urls"])):
            raise ValueError("invalid unsettled policy publication")
        policy_version(publication["version"])
        for name in ("operation_id", "model_configuration_fingerprint", "publisher_id"):
            nonempty(publication[name], name)
    barrier = state["barrier"]
    if barrier is not None:
        expected = {"operation_id", "group_id", "identity", "source_leases", "deadline_unix", "state"}
        if (not isinstance(barrier, dict) or set(barrier) != expected
                or barrier["state"] not in {"ACTIVE", "BLOCKED"} or not barrier["source_leases"]
                or _decode_policy_identity(barrier["identity"]).policy_version != active):
            raise ValueError("invalid persisted handoff publication barrier")
        for name in ("operation_id", "group_id"):
            nonempty(barrier[name], name)
        if not math.isfinite(barrier["deadline_unix"]):
            raise ValueError("invalid handoff publication deadline")
        for lease in barrier["source_leases"]:
            if (set(lease) != {"trajectory_id", "owner_id", "generation", "lease_id"}
                    or type(lease["generation"]) is not int or lease["generation"] < 0):
                raise ValueError("invalid handoff source execution identity")


class PolicyPublicationDriver:
    """Call original transfer and admission hooks once; uncertain writes block.

    ``update_checkpoint(version)`` is the original transfer stage with admission
    already stopped. ``observe_engines`` performs real /model_info reads while
    the engines remain isolated. It must not manufacture expected versions from
    configuration. The root integration owns these callbacks and their URLs.
    """

    def __init__(self, client):
        self.client = client

    async def prepare(self, *, identity: dict, model_configuration_fingerprint: str,
                      publisher_id: str, engine_urls: list[str]) -> PolicyPublicationTicket:
        """Reserve before the original lifecycle starts any parameter mutation."""
        return await self.client.begin_policy_publication(
            identity, model_configuration_fingerprint, publisher_id, engine_urls)

    async def mark_update_started(self, ticket: PolicyPublicationTicket) -> None:
        await self.client.mark_policy_update_started(ticket)

    async def confirm_before_admission(self, ticket: PolicyPublicationTicket,
                                       observe_engines: Callable[[], Awaitable[list[dict]]]) -> PublishedPolicyRecord:
        """After original transfer ACKs, before original Gateway registration.

        Read owned server handles or owned URL /model_info directly. A Gateway
        reader which requires registered workers cannot be used at this point.
        """
        actual = await observe_engines()
        return PublishedPolicyRecord.decode(await self.client.confirm_policy_publication(ticket, actual))

    async def fail(self, ticket: PolicyPublicationTicket, reason: str) -> None:
        await self.client.fail_policy_publication(ticket, reason)

    async def publish(self, *, identity: dict, model_configuration_fingerprint: str,
                      publisher_id: str, engine_urls: list[str],
                      update_checkpoint: Callable[[int], Awaitable[Any]],
                      observe_engines: Callable[[], Awaitable[list[dict]]],
                      stop_admission: Callable[[], Awaitable[Any]],
                      start_admission: Callable[[int], Awaitable[Any]]) -> PublishedPolicyRecord:
        canonical = _decode_policy_identity(identity)
        version = int(policy_version(canonical.policy_version))
        ticket = await self.prepare(identity=identity, model_configuration_fingerprint=model_configuration_fingerprint,
                                    publisher_id=publisher_id, engine_urls=engine_urls)
        committed = False
        try:
            await stop_admission()
            await self.mark_update_started(ticket)
            await update_checkpoint(version)
            record = await self.confirm_before_admission(ticket, observe_engines)
            committed = True
            await start_admission(version)
            return record
        except BaseException as error:
            # Cancellation is not evidence that an external write stopped. No
            # finally block reopens admission or frees publication ownership.
            if not committed:
                try:
                    await asyncio.shield(self.client.fail_policy_publication(ticket, type(error).__name__))
                except BaseException:
                    pass  # Preserve the direct error; persisted intent remains unsettled.
            raise
