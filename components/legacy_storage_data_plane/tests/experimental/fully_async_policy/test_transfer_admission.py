from verl.experimental.fully_async_policy.transfer_admission import (
    AdmissionAction,
    AdmissionReason,
    TransferAdmissionConfig,
    TransferAdmissionController,
    TransferIntent,
    TransferRequest,
)


def request(name, intent, size=10, **kwargs):
    return TransferRequest(transfer_id=name, intent=intent, bytes=size, **kwargs)


def test_shadow_mode_records_pressure_without_blocking_real_transfer():
    controller = TransferAdmissionController(
        TransferAdmissionConfig(
            shadow=True,
            max_inflight_transfers=1,
            max_inflight_bytes=100,
            max_background_inflight=1,
        )
    )
    first = controller.begin(request("weight", TransferIntent.WEIGHT_SYNC, 60))
    second = controller.begin(request("artifact", TransferIntent.ROLLOUT_ARTIFACT, 20))

    assert first.action is AdmissionAction.ADMIT
    assert second.action is AdmissionAction.QUEUE
    assert second.reason is AdmissionReason.TRANSFER_SLOT_PRESSURE
    assert second.affects_execution is False
    assert controller.summary()["transfer_admission/active"] == 2.0

    controller.complete("weight")
    controller.complete("artifact")
    assert controller.summary()["transfer_admission/active"] == 0.0


def test_remote_restore_requires_cost_gate_approval_even_in_shadow():
    controller = TransferAdmissionController()
    denied = controller.evaluate(
        request(
            "restore-denied",
            TransferIntent.REMOTE_KV_RESTORE,
            cost_gate_approved=False,
        )
    )
    approved = controller.evaluate(
        request(
            "restore-approved",
            TransferIntent.REMOTE_KV_RESTORE,
            cost_gate_approved=True,
        )
    )

    assert denied.action is AdmissionAction.REJECT
    assert denied.reason is AdmissionReason.RESTORE_COST_GATE_DENIED
    assert approved.action is AdmissionAction.ADMIT


def test_background_budget_preserves_capacity_for_foreground_intents():
    controller = TransferAdmissionController(
        TransferAdmissionConfig(
            max_inflight_transfers=3,
            max_inflight_bytes=100,
            max_background_inflight=1,
        )
    )
    controller.begin(request("prefetch-0", TransferIntent.PREFETCH, 10))
    queued = controller.evaluate(request("checkpoint-0", TransferIntent.CHECKPOINT, 10))
    foreground = controller.evaluate(request("artifact-0", TransferIntent.ROLLOUT_ARTIFACT, 10))

    assert queued.action is AdmissionAction.QUEUE
    assert queued.reason is AdmissionReason.BACKGROUND_BUDGET_EXHAUSTED
    assert foreground.action is AdmissionAction.ADMIT


def test_enforcing_mode_only_marks_admitted_transfer_active():
    controller = TransferAdmissionController(
        TransferAdmissionConfig(
            shadow=False,
            max_inflight_transfers=1,
            max_inflight_bytes=100,
            max_background_inflight=1,
        )
    )
    admitted = controller.begin(request("weight", TransferIntent.WEIGHT_SYNC, 60))
    queued = controller.begin(request("artifact", TransferIntent.ROLLOUT_ARTIFACT, 20))

    assert admitted.affects_execution is True
    assert queued.action is AdmissionAction.QUEUE
    assert controller.summary()["transfer_admission/active"] == 1.0


def test_oversized_transfer_is_rejected_and_invalid_config_fails_closed():
    controller = TransferAdmissionController(
        TransferAdmissionConfig(max_inflight_bytes=100)
    )
    decision = controller.evaluate(request("huge", TransferIntent.WEIGHT_SYNC, 101))
    assert decision.action is AdmissionAction.REJECT
    assert decision.reason is AdmissionReason.REQUEST_TOO_LARGE

    try:
        TransferAdmissionConfig(max_inflight_transfers=1, max_background_inflight=2)
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
    else:
        raise AssertionError("invalid transfer admission config must fail")
