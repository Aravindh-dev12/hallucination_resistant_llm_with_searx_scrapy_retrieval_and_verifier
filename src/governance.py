"""Governance primitives for training resources and OEON kernel services.

Policy is enforced in code. Prompts and model output never grant authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Mapping, Sequence


class GovernanceError(RuntimeError):
    """Raised when a candidate attempts to cross a governance boundary."""


class ResourceRole(str, Enum):
    TRAINING_RESOURCE = "training_resource"
    PERCEPTION_SERVICE = "perception_service"
    WORLD_STATE_SERVICE = "world_state_service"
    KERNEL_SERVICE = "kernel_service"
    COORDINATOR = "coordinator"


class AdmissionDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class ResourceRecord:
    name: str
    role: ResourceRole
    version: str
    source_uri: str
    license_id: str
    content_digest: str
    provenance_accepted: bool = False
    verifier_receipts: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    parent_checkpoint: str
    proposed_by: str
    resource_digests: tuple[str, ...]
    evaluation_receipts: tuple[str, ...]
    consent_receipt: str | None = None


@dataclass(frozen=True)
class AuditReceipt:
    event: str
    actor: str
    subject: str
    decision: AdmissionDecision
    reasons: tuple[str, ...]
    timestamp: str
    previous_digest: str | None = None
    digest: str = field(default="", compare=False)

    @classmethod
    def issue(
        cls,
        *,
        event: str,
        actor: str,
        subject: str,
        decision: AdmissionDecision,
        reasons: Sequence[str],
        previous_digest: str | None = None,
    ) -> "AuditReceipt":
        payload = {
            "event": event,
            "actor": actor,
            "subject": subject,
            "decision": decision.value,
            "reasons": list(reasons),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "previous_digest": previous_digest,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(**payload, decision=decision, reasons=tuple(reasons), digest=digest)


@dataclass(frozen=True)
class GovernancePolicy:
    compatible_licenses: frozenset[str]
    required_verifiers: int = 1
    promotion_quorum: int = 2
    require_consent: bool = True

    def assess_training_resource(self, resource: ResourceRecord) -> AuditReceipt:
        reasons: list[str] = []
        if resource.role is not ResourceRole.TRAINING_RESOURCE:
            reasons.append("resource is not classified as training-only")
        if not resource.provenance_accepted:
            reasons.append("provenance has not been accepted")
        if resource.license_id not in self.compatible_licenses:
            reasons.append(f"incompatible or unknown license: {resource.license_id}")
        if len(resource.verifier_receipts) < self.required_verifiers:
            reasons.append("insufficient verifier-backed receipts")
        if not resource.content_digest:
            reasons.append("missing immutable content digest")
        decision = AdmissionDecision.REJECT if reasons else AdmissionDecision.ACCEPT
        return AuditReceipt.issue(
            event="training_resource_admission",
            actor="oeon-governance",
            subject=resource.name,
            decision=decision,
            reasons=reasons or ["all admission gates passed"],
        )


class KernelGate:
    """The only component allowed to authorize candidate promotion."""

    def __init__(self, policy: GovernancePolicy):
        self.policy = policy

    def authorize_promotion(
        self,
        candidate: CandidateRecord,
        *,
        approved_resources: Mapping[str, ResourceRecord],
        independent_approvals: Iterable[str],
        sandbox_receipt: str | None,
        formal_verification_receipt: str | None,
    ) -> AuditReceipt:
        reasons: list[str] = []
        approvals = set(independent_approvals)

        if candidate.proposed_by.lower() == "tinton" and "tinton" in {
            item.lower() for item in approvals
        }:
            reasons.append("Tinton cannot approve its own candidate")
        if len({a.lower() for a in approvals if a.lower() != "tinton"}) < self.policy.promotion_quorum:
            reasons.append("independent promotion quorum not met")
        if self.policy.require_consent and not candidate.consent_receipt:
            reasons.append("missing consent receipt")
        if not sandbox_receipt:
            reasons.append("missing sandbox execution receipt")
        if not formal_verification_receipt:
            reasons.append("missing formal verification receipt")

        for digest in candidate.resource_digests:
            resource = approved_resources.get(digest)
            if resource is None:
                reasons.append(f"unapproved training resource digest: {digest}")
                continue
            admission = self.policy.assess_training_resource(resource)
            if admission.decision is not AdmissionDecision.ACCEPT:
                reasons.append(f"resource rejected: {resource.name}")

        decision = AdmissionDecision.REJECT if reasons else AdmissionDecision.ACCEPT
        return AuditReceipt.issue(
            event="checkpoint_promotion",
            actor="oeon-kernel",
            subject=candidate.candidate_id,
            decision=decision,
            reasons=reasons or ["kernel gates and independent quorum passed"],
        )


class TintonCoordinator:
    """Coordinates candidates; deliberately exposes no authorization method."""

    name = "tinton"

    def propose(
        self,
        *,
        candidate_id: str,
        parent_checkpoint: str,
        resource_digests: Sequence[str],
        evaluation_receipts: Sequence[str],
        consent_receipt: str | None,
    ) -> CandidateRecord:
        return CandidateRecord(
            candidate_id=candidate_id,
            parent_checkpoint=parent_checkpoint,
            proposed_by=self.name,
            resource_digests=tuple(resource_digests),
            evaluation_receipts=tuple(evaluation_receipts),
            consent_receipt=consent_receipt,
        )

    def promote(self, *_args, **_kwargs) -> None:
        raise GovernanceError(
            "Tinton cannot authorize, bypass kernel gates, or promote a checkpoint"
        )
