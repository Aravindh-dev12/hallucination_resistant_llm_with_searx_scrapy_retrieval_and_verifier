from citation_extractor import citations_from_claims
from claim_verification import verification_summary, verify_claims
from instruct_generator import InstructGenerator
from reranker import Reranker
from retrieval_security import detect_prompt_injection, sanitize_evidence
from retriever import Retriever
from utils import load_config
from verifier_ensemble import VerifierEnsemble

cfg = load_config()


class Pipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.generator = InstructGenerator()
        self.verifier = VerifierEnsemble()

    def _safe_evidence(self, documents):
        safe = []
        for document in documents:
            text = sanitize_evidence(str(document.get("text", "")))
            if not text or detect_prompt_injection(text):
                continue
            safe.append({**document, "text": text})
        return safe

    def _verify(self, answer, evidence):
        claims = verify_claims(
            answer,
            evidence,
            self.verifier.classify,
            entailment_threshold=cfg["verification"]["entailment_threshold"],
            contradiction_threshold=cfg["verification"]["contradiction_threshold"],
        )
        return claims, verification_summary(claims)

    def _abstention(self, claims):
        supported = [
            result["claim"]
            for result in claims
            if result["status"] == "supported"
        ]
        if supported:
            return (
                "I could verify only the following statements:\n\n- "
                + "\n- ".join(supported)
                + "\n\nOther generated claims were withheld because the evidence "
                "was insufficient or contradictory."
            )
        return (
            "I could not produce an answer that was sufficiently supported by "
            "the retrieved evidence."
        )

    def answer(self, query):
        documents, retrieval_scores = self.retriever.retrieve(
            query, k=cfg["retrieval"]["candidate_k"]
        )
        documents = self._safe_evidence(documents)
        if not documents:
            return {
                "query": query,
                "answer": self._abstention([]),
                "status": "abstained",
                "verifier_score": 0.0,
                "verification": verification_summary([]),
                "claims": [],
                "citations": [],
                "flagged": True,
            }

        reranked, rerank_scores = self.reranker.rerank(
            query, documents, top_k=cfg["retrieval"]["rerank_top_k"]
        )
        evidence = reranked[: cfg["retrieval"]["evidence_k"]]
        context = "\n\n".join(
            f"[Source {index}] {document['text']}"
            for index, document in enumerate(evidence)
        )
        answer = self.generator.generate(
            query,
            context=context,
            max_tokens=cfg["inference"]["max_tokens"],
            temperature=cfg["inference"]["temperature"],
        )
        claims, summary = self._verify(answer, evidence)

        if not summary["all_supported"] and cfg["verification"]["corrective_retries"] > 0:
            unsupported = [
                item["claim"]
                for item in claims
                if item["status"] != "supported"
            ]
            correction_prompt = (
                f"Answer this question using only the supplied context: {query}\n"
                "Rewrite the draft so every factual claim is directly supported. "
                "Remove these unsupported or contradictory claims:\n- "
                + "\n- ".join(unsupported)
            )
            corrected = self.generator.generate(
                correction_prompt,
                context=context,
                max_tokens=cfg["inference"]["max_tokens"],
                temperature=0.0,
            )
            corrected_claims, corrected_summary = self._verify(corrected, evidence)
            if corrected_summary["faithfulness"] >= summary["faithfulness"]:
                answer, claims, summary = (
                    corrected,
                    corrected_claims,
                    corrected_summary,
                )

        status = "verified" if summary["all_supported"] else "abstained"
        if status == "abstained":
            answer = self._abstention(claims)

        return {
            "query": query,
            "answer": answer,
            "status": status,
            "verifier_score": summary["faithfulness"],
            "verification": summary,
            "claims": claims,
            "citations": citations_from_claims(claims),
            "retrieval_scores": retrieval_scores,
            "rerank_scores": [float(score) for score in rerank_scores],
            "flagged": status != "verified",
        }
