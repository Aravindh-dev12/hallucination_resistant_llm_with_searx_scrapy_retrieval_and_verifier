import pytest

from governance import (
    AdmissionDecision,
    GovernanceError,
    GovernancePolicy,
    KernelGate,
    ResourceRecord,
    ResourceRole,
    TintonCoordinator,
)


def policy():
    return GovernancePolicy(
        compatible_licenses=frozenset({"MIT", "Apache-2.0"}),
        required_verifiers=1,
        promotion_quorum=2,
    )


def accepted_resource():
    return ResourceRecord(
        name="VibeThinker",
        role=ResourceRole.TRAINING_RESOURCE,
        version="pinned-version",
        source_uri="https://github.com/WeiboAI/VibeThinker",
        license_id="MIT",
        content_digest="sha256:abc",
        provenance_accepted=True,
        verifier_receipts=("verifier:1",),
    )


def test_training_resource_requires_provenance_license_and_verifier():
    rejected = ResourceRecord(
        name="unknown",
        role=ResourceRole.TRAINING_RESOURCE,
        version="latest",
        source_uri="https://example.invalid",
        license_id="UNKNOWN",
        content_digest="",
    )
    receipt = policy().assess_training_resource(rejected)
    assert receipt.decision is AdmissionDecision.REJECT
    assert len(receipt.reasons) >= 3


def test_tinton_cannot_promote():
    with pytest.raises(GovernanceError):
        TintonCoordinator().promote("candidate")


def test_tinton_cannot_self_approve():
    resource = accepted_resource()
    candidate = TintonCoordinator().propose(
        candidate_id="candidate-1",
        parent_checkpoint="sovereign@sha256:parent",
        resource_digests=(resource.content_digest,),
        evaluation_receipts=("eval:1",),
        consent_receipt="consent:1",
    )
    receipt = KernelGate(policy()).authorize_promotion(
        candidate,
        approved_resources={resource.content_digest: resource},
        independent_approvals={"Tinton", "Verifier-A"},
        sandbox_receipt="sandbox:1",
        formal_verification_receipt="formal:1",
    )
    assert receipt.decision is AdmissionDecision.REJECT
    assert any("cannot approve" in reason for reason in receipt.reasons)


def test_kernel_can_promote_after_all_independent_gates_pass():
    resource = accepted_resource()
    candidate = TintonCoordinator().propose(
        candidate_id="candidate-2",
        parent_checkpoint="sovereign@sha256:parent",
        resource_digests=(resource.content_digest,),
        evaluation_receipts=("eval:1",),
        consent_receipt="consent:1",
    )
    receipt = KernelGate(policy()).authorize_promotion(
        candidate,
        approved_resources={resource.content_digest: resource},
        independent_approvals={"Verifier-A", "Verifier-B"},
        sandbox_receipt="sandbox:1",
        formal_verification_receipt="formal:1",
    )
    assert receipt.decision is AdmissionDecision.ACCEPT
    assert receipt.digest
