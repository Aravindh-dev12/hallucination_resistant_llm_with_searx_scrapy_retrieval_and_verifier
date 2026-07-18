from claim_verification import (
    extract_atomic_claims,
    verification_summary,
    verify_claims,
)


def classifier(evidence, claim):
    if "Einstein" in evidence and "relativity" in claim:
        return {"entailment": 0.95, "contradiction": 0.01}
    if "not born in France" in evidence and "born in France" in claim:
        return {"entailment": 0.01, "contradiction": 0.93}
    return {"entailment": 0.1, "contradiction": 0.1}


def test_claims_are_verified_independently():
    answer = "Einstein developed relativity. Einstein was born in France."
    evidence = [
        {"text": "Einstein developed the theory of relativity.", "url": "a"},
        {"text": "Einstein was not born in France.", "url": "b"},
    ]
    results = verify_claims(answer, evidence, classifier)
    assert [result["status"] for result in results] == [
        "supported",
        "contradicted",
    ]
    summary = verification_summary(results)
    assert summary["faithfulness"] == 0.5
    assert not summary["all_supported"]


def test_empty_or_nonfactual_output_does_not_pass():
    assert extract_atomic_claims("I don't know.") == []
    assert not verification_summary([])["all_supported"]
