from verl.experimental.fully_async_policy.kv_export_proof import confirm_export


def test_only_completed_actual_objects_make_a_kv_reference():
    objects = {"page-k": b"actual k", "page-v": b"actual v"}
    kwargs = dict(token_ids=[1, 2, 3], prefix_tokens=2, policy_identity={"version": "0"},
        ordered_object_keys=list(objects), read_object=objects.get, master_address="node:50051",
        source_completed_tokens=2)
    ref = confirm_export(**kwargs)
    assert ref["kv_bytes"] == sum(map(len, objects.values()))
    assert ref["prefix_tokens"] == 2 and ref["export_complete"]
    assert all(len(item["sha256"]) == 64 for item in ref["storage_reference"]["objects"])
    objects.pop("page-v")
    assert confirm_export(**kwargs) is None


def test_candidate_keys_or_incomplete_source_do_not_claim_completed_export():
    kwargs = dict(token_ids=[1, 2, 3], prefix_tokens=2, policy_identity={"version": "0"},
        ordered_object_keys=["page-k", "page-v"], read_object=lambda _: b"candidate",
        master_address="node:50051", source_completed_tokens=None)
    assert confirm_export(**kwargs) is None
    assert confirm_export(**{**kwargs, "source_completed_tokens": 1}) is None


def test_failed_get_keeps_token_fallback_available():
    def failed(_):
        raise RuntimeError("store is unavailable")
    assert confirm_export(token_ids=[1, 2], prefix_tokens=1, policy_identity={},
        ordered_object_keys=["k"], read_object=failed, master_address="node:50051",
        source_completed_tokens=1) is None


def test_candidate_handle_and_invented_bytes_are_rejected():
    import pytest
    from copy import deepcopy
    from verl.experimental.fully_async_policy.kv_export_proof import validate_export_reference
    ref = confirm_export(token_ids=[1, 2], prefix_tokens=1, policy_identity={"version": "0"},
        ordered_object_keys=["k"], read_object=lambda _: b"real kv", master_address="store:50051",
        source_completed_tokens=1)
    validate_export_reference(ref, token_ids=[1, 2], policy_identity={"version": "0"})
    bad = deepcopy(ref)
    bad["kv_bytes"] += 1
    with pytest.raises(ValueError, match="bytes"):
        validate_export_reference(bad, token_ids=[1, 2], policy_identity={"version": "0"})
    bad["storage_reference"] = "candidate handle"
    with pytest.raises(ValueError):
        validate_export_reference(bad, token_ids=[1, 2], policy_identity={"version": "0"})
