"""Atomic-claim verification and evidence binding."""
from __future__ import annotations

import re
from typing import Callable, Mapping, Sequence

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
NON_CLAIM_PATTERNS = (
    re.compile(r"^(?:i\s+)?(?:do not|don't|cannot|can't)\s+(?:know|verify|confirm)", re.I),
    re.compile(r"^(?:insufficient|not enough)\s+evidence", re.I),
)


def extract_atomic_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for sentence in SENTENCE_RE.split(answer.strip()):
        sentence = sentence.strip(" \t-*")
        if (
            not sentence
            or len(sentence.split()) < 3
            or any(pattern.search(sentence) for pattern in NON_CLAIM_PATTERNS)
        ):
            continue
        parts = re.split(r"\s*;\s*|\s+and\s+(?=[A-Z0-9])", sentence)
        claims.extend(part.strip() for part in parts if len(part.split()) >= 3)
    return claims


def verify_claims(
    answer: str,
    evidence: Sequence[Mapping[str, object]],
    classify: Callable[[str, str], Mapping[str, float]],
    *,
    entailment_threshold: float = 0.72,
    contradiction_threshold: float = 0.60,
) -> list[dict]:
    results = []
    for claim_id, claim in enumerate(extract_atomic_claims(answer), start=1):
        candidates = []
        for evidence_id, document in enumerate(evidence):
            scores = dict(classify(str(document.get("text", "")), claim))
            candidates.append((evidence_id, document, scores))

        strongest_contradiction = max(
            candidates,
            key=lambda item: item[2].get("contradiction", 0.0),
            default=None,
        )
        strongest_support = max(
            candidates,
            key=lambda item: item[2].get("entailment", 0.0),
            default=None,
        )

        contradiction_score = (
            float(strongest_contradiction[2].get("contradiction", 0.0))
            if strongest_contradiction
            else 0.0
        )
        entailment_score = (
            float(strongest_support[2].get("entailment", 0.0))
            if strongest_support
            else 0.0
        )

        if contradiction_score >= contradiction_threshold:
            evidence_id, document, _scores = strongest_contradiction
            status, score = "contradicted", contradiction_score
        elif entailment_score >= entailment_threshold:
            evidence_id, document, _scores = strongest_support
            status, score = "supported", entailment_score
        elif strongest_support:
            evidence_id, document, _scores = strongest_support
            status, score = "insufficient", entailment_score
        else:
            status, score, evidence_id, document = (
                "insufficient",
                0.0,
                None,
                {},
            )

        results.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "status": status,
                "score": score,
                "evidence_id": evidence_id,
                "source": document.get("url") or document.get("source"),
                "passage": str(document.get("text", ""))[:500],
            }
        )
    return results


def verification_summary(results: Sequence[Mapping[str, object]]) -> dict:
    total = len(results)
    supported = sum(item.get("status") == "supported" for item in results)
    contradicted = sum(item.get("status") == "contradicted" for item in results)
    return {
        "total_claims": total,
        "supported_claims": supported,
        "contradicted_claims": contradicted,
        "faithfulness": supported / total if total else 0.0,
        "all_supported": total > 0 and supported == total,
    }
