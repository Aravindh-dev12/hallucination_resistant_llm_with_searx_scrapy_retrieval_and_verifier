"""Build citations only from verified claim-to-evidence bindings."""


def citations_from_claims(claim_results):
    citations = []
    seen = set()
    for result in claim_results:
        if result.get("status") != "supported":
            continue
        key = (result.get("claim_id"), result.get("evidence_id"))
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "claim_id": result.get("claim_id"),
                "idx": result.get("evidence_id"),
                "source": result.get("source"),
                "snippet": result.get("passage", ""),
                "entailment_score": result.get("score", 0.0),
            }
        )
    return citations


def extract_citations(_answer, _evidence_list):
    raise RuntimeError(
        "string-matching citations were removed; use citations_from_claims"
    )
