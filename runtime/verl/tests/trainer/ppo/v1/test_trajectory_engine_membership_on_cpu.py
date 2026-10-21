# Copyright 2026 Project B contributors
# Licensed under the Apache License, Version 2.0.
"""Replica membership is distinct from a new optimizer/weight publication."""

import pytest
from test_trajectory_checkpoint_storage_on_cpu import publish, state


def observation(url, version=0):
    return {"engine_url": url, "weight_version": str(version), "source": "model_info"}


def test_engine_join_preserves_policy_identity_and_does_not_allow_duplicate_update(tmp_path):
    coordinator = state.DurableTrajectoryCoordinator(str(tmp_path), "run")
    previous = publish(coordinator, 0)
    try:
        joined = coordinator.authorize_same_version_engines(0, ["http://borrowed"], [observation("http://borrowed")])
        for field in ["run_id", "identity", "publisher_id", "model_configuration_fingerprint", "committed_at_unix"]:
            assert joined[field] == previous[field]
        assert {row["engine_url"] for row in joined["engine_observations"]} == {"http://fixture", "http://borrowed"}
        with pytest.raises(state.PolicyPublicationBlocked, match="already published"):
            publish(coordinator, 0)
    finally:
        coordinator.close()
    reopened = state.DurableTrajectoryCoordinator(str(tmp_path), "run")
    try:
        assert (state.PublishedPolicyRecord.decode(reopened.get_active_published_policy())
                == state.PublishedPolicyRecord.decode(joined))
        next_version = publish(reopened, 1)
        assert next_version["identity"]["policy_version"] == "1"
        assert [row["engine_url"] for row in next_version["engine_observations"]] == ["http://fixture"]
    finally:
        reopened.close()


@pytest.mark.parametrize("fault", ["wrong_version", "missing", "duplicate", "unexpected", "unsettled"])
def test_unverified_engine_is_never_authorized(tmp_path, fault):
    coordinator = state.DurableTrajectoryCoordinator(str(tmp_path), "run")
    previous = publish(coordinator, 0)
    urls, rows, version = ["http://borrowed"], [observation("http://borrowed")], 0
    if fault == "wrong_version":
        rows = [observation("http://borrowed", 1)]
    elif fault == "missing":
        rows = []
    elif fault == "duplicate":
        rows *= 2
    elif fault == "unexpected":
        rows = [observation("http://foreign")]
    else:
        identity = {**previous["identity"], "policy_version": "1", "model_weights_fingerprint": "weights-1"}
        coordinator.begin_policy_publication(identity, "fixture-config", "fixture-checkpoint", ["http://fixture"])
    try:
        with pytest.raises((ValueError, state.PolicyPublicationBlocked)):
            coordinator.authorize_same_version_engines(version, urls, rows)
        assert coordinator.get_published_policy(0) == previous
    finally:
        coordinator.close()
