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

"""Exact protocol migration from Project B f91c201d; see MIGRATION.md."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CanonicalPolicyKVIdentity:
    """The cross-replica identity for a policy and its KV representation.

    Replica-local allocator generations and request handles are deliberately
    excluded. They validate local runtime state, not whether two replicas can
    safely interpret the same KV blocks.
    """

    policy_version: str
    model_weights_fingerprint: str
    tokenizer_template_fingerprint: str
    kv_cache_schema_fingerprint: str

    @classmethod
    def for_fully_async_policy_version(cls, policy_version: int | str) -> "CanonicalPolicyKVIdentity":
        """Legacy test helper; runtime code must use the runtime-config builder."""
        version = str(policy_version)
        return cls(
            policy_version=version,
            model_weights_fingerprint=_canonical_fingerprint(
                {"legacy_test_policy_version": version}
            ),
            tokenizer_template_fingerprint=_canonical_fingerprint(
                {"legacy_test_tokenizer_template": "fully_async"}
            ),
            kv_cache_schema_fingerprint=_canonical_fingerprint(
                {"legacy_test_kv_schema": "fully_async"}
            ),
        )

    def validate(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

    def is_complete(self) -> bool:
        try:
            self.validate()
        except ValueError:
            return False
        return True

    @property
    def digest(self) -> str:
        self.validate()
        canonical_identity = json.dumps(vars(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()

    @property
    def cache_salt(self) -> str:
        """The exact digest passed to SGLang's existing cache namespace."""
        return self.digest


def _decode_policy_identity(payload: Any) -> CanonicalPolicyKVIdentity:
    if not isinstance(payload, dict) or set(payload) != {
        "policy_version",
        "model_weights_fingerprint",
        "tokenizer_template_fingerprint",
        "kv_cache_schema_fingerprint",
    }:
        raise ValueError("policy identity snapshot has an invalid schema")
    if any(not isinstance(value, str) for value in payload.values()):
        raise ValueError("policy identity snapshot fields must be strings")
    identity = CanonicalPolicyKVIdentity(**payload)
    identity.validate()
    return identity
