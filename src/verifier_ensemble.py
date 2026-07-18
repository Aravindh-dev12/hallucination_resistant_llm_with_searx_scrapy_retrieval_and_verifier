from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from utils import load_config

cfg = load_config()
V1 = cfg["model"]["verifier_1"]
V2 = cfg["model"]["verifier_2"]
device = "cuda" if torch.cuda.is_available() else "cpu"


class VerifierEnsemble:
    def __init__(self):
        self.tok1 = AutoTokenizer.from_pretrained(V1, use_fast=True)
        self.m1 = AutoModelForSequenceClassification.from_pretrained(V1).to(device)
        self.tok2 = AutoTokenizer.from_pretrained(V2, use_fast=True)
        self.m2 = AutoModelForSequenceClassification.from_pretrained(V2).to(device)

    @staticmethod
    def _labels(model, probabilities):
        labels = {
            str(label).lower(): float(probabilities[index])
            for index, label in model.config.id2label.items()
        }

        def find(name, fallback):
            return next(
                (score for label, score in labels.items() if name in label),
                fallback,
            )

        return {
            "contradiction": find("contradiction", float(probabilities[0])),
            "neutral": find(
                "neutral",
                float(probabilities[1]) if len(probabilities) > 2 else 0.0,
            ),
            "entailment": find("entailment", float(probabilities[-1])),
        }

    def _classify_one(self, tokenizer, model, premise, hypothesis):
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
        return self._labels(model, probabilities.detach().cpu().tolist())

    def classify(self, premise, hypothesis):
        first = self._classify_one(self.tok1, self.m1, premise, hypothesis)
        second = self._classify_one(self.tok2, self.m2, premise, hypothesis)
        return {
            label: (first[label] + second[label]) / 2.0
            for label in ("contradiction", "neutral", "entailment")
        }

    def entail_score(self, premise, hypothesis):
        return self.classify(premise, hypothesis)["entailment"]

    def score(self, claim, evidence_list):
        scores = [self.entail_score(evidence, claim) for evidence in evidence_list]
        # A claim needs one supporting passage; unrelated passages must not dilute it.
        return (max(scores) if scores else 0.0), scores
