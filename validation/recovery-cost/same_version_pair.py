import copy, hashlib, json, math
_STATUS = "PREPARED_REQUEST_ARGUMENTS_NOT_EXECUTED"
_SCOPE = "SAME_VERSION_FIXED_BOUNDARY_REQUEST_ONLY"
def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
def _ids(seq):
    return (isinstance(seq, list) and seq and all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in seq))
def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
def _get(d, key, where):
    if not isinstance(d, dict) or key not in d:
        raise ValueError("missing %s in %s" % (key, where))
    return d[key]
def same_version_requests(snapshot, target_identity, fresh_target_request_id):
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be dict")
    if _get(snapshot, "schema_version", "snapshot") != 2:
        raise ValueError("unsupported snapshot schema")
    batches = _get(snapshot, "completed_tool_batches", "snapshot")
    if not (isinstance(batches, list) and batches):
        raise ValueError("completed_tool_batches must be nonempty")
    ref = _get(snapshot, "kv_ref", "snapshot")
    if _get(ref, "snapshot_committed", "kv_ref") is not True:
        raise ValueError("snapshot not committed")
    policy = _get(snapshot, "policy_identity", "snapshot")
    if not isinstance(policy, dict) or policy != _get(ref, "policy_identity", "kv_ref"):
        raise ValueError("policy identity mismatch")
    if ref.get("identity") != target_identity:
        raise ValueError("kv ref does not target requested identity")
    pver = str(_get(policy, "policy_version", "policy_identity"))
    for ident in (target_identity, _get(ref, "identity", "kv_ref")):
        if str(_get(ident, "actualweight_version", "identity")) != pver \
           or str(_get(ident, "weight_version", "identity")) != pver:
            raise ValueError("cross-weight reuse unsupported")
        if _get(ident, "tp_size", "identity") != 2:
            raise ValueError("TP2 required")
    extra = _get(_get(snapshot, "agent_state", "snapshot"), "extra_fields", "agent_state")
    frame = _get(extra, "trajectory_inflight_generation", "extra_fields")
    iid, oid = _get(frame, "input_ids", "frame"), _get(frame, "output_ids", "frame")
    if not (_ids(iid) and _ids(oid)):
        raise ValueError("invalid token ids")
    lp = _get(frame, "output_logprobs", "frame")
    if not (isinstance(lp, list) and len(lp) == len(oid) and all(_num(v) for v in lp)):
        raise ValueError("logprobs length/type mismatch")
    orig, rem = _get(frame, "original_max_new_tokens", "frame"), _get(frame, "remaining_generation_tokens", "frame")
    if not (isinstance(orig, int) and not isinstance(orig, bool) and orig > 0 and
            isinstance(rem, int) and not isinstance(rem, bool) and rem > 0 and rem == orig - len(oid)):
        raise ValueError("invalid generation budget")
    sp = _get(frame, "requested_sampling_params", "frame")
    if not isinstance(sp, dict):
        raise ValueError("invalid sampling params")
    lrid = _get(frame, "logical_request_id", "frame")
    srid = _get(frame, "source_engine_request_id", "frame")
    if not all(isinstance(x, str) and x for x in (lrid, srid)): raise ValueError("missing source/logical request id")
    if not (isinstance(fresh_target_request_id, str) and fresh_target_request_id
            and fresh_target_request_id != srid):
        raise ValueError("invalid fresh target request id")
    sp_t = copy.deepcopy(sp)
    sp_t.pop("max_tokens", None); sp_t.pop("max_new_tokens", None)
    sp_t["max_tokens"] = rem
    trace = {"input_ids": iid, "output_ids": oid, "output_logprobs": lp,
             "completed_tool_batches": batches, "requested_sampling_params": sp,
             "remaining_generation_tokens": rem, "original_max_new_tokens": orig,
             "policy_identity": policy, "tool_version": _get(snapshot, "tool_version", "snapshot"),
             "environment_version": _get(snapshot, "environment_version", "snapshot")}
    return {"status": _STATUS, "scope": _SCOPE,
            "A_kwargs": {"request_id": lrid, "prompt_ids": list(iid) + list(oid),
                         "sampling_params": sp_t},
            "B_kwargs": {"ref": copy.deepcopy(ref), "request_id": fresh_target_request_id,
                         "limit": orig, "observed_identity": copy.deepcopy(target_identity)},
            "trace_sha256": _sha(trace), "completed_tools_sha256": _sha(batches),
            "sampling_sha256": _sha(sp),
            "suffix_equality_requirement": ("GREEDY_FIXED_RUN_CHECK_ONLY"
                if sp.get("temperature") == 0 else "FUNCTIONAL_BUDGET_AND_QUALITY_ONLY")}
