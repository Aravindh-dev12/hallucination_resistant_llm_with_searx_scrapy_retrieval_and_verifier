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
        best = max(
            candidates,
            key=lambda item: (
                item[2].get("entailment", 0.0)
                - item[2].get("contradiction", 0.0)
            ),
            default=None,
        )
        if best is None:
            status, score, evidence_id, document = (
                "insufficient",
                0.0,
                None,
                {},
            )
        else:
            evidence_id, document, scores = best
            entailment = float(scores.get("entailment", 0.0))
            contradiction = float(scores.get("contradiction", 0.0))
            if contradiction >= contradiction_threshold:
                status, score = "contradicted", contradiction
            elif entailment >= entailment_threshold:
                status, score = "supported", entailment
            else:
                status, score = "insufficient", entailment
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
