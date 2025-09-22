from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from utils import load_config

cfg = load_config()
V1 = cfg['model']['verifier_1']
V2 = cfg['model']['verifier_2']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VerifierEnsemble:
    def __init__(self):
        self.tok1 = AutoTokenizer.from_pretrained(V1, use_fast=True)
        self.m1 = AutoModelForSequenceClassification.from_pretrained(V1).to(device)
        self.tok2 = AutoTokenizer.from_pretrained(V2, use_fast=True)
        self.m2 = AutoModelForSequenceClassification.from_pretrained(V2).to(device)
    def entail_score(self, premise, hypothesis):
        inp1 = self.tok1(premise, hypothesis, return_tensors='pt', truncation=True).to(device)
        o1 = torch.softmax(self.m1(**inp1).logits, dim=-1).detach().cpu().numpy()[0]
        e1 = float(o1[2]) if o1.shape[0] > 2 else float(o1[-1])
        inp2 = self.tok2(premise, hypothesis, return_tensors='pt', truncation=True).to(device)
        o2 = torch.softmax(self.m2(**inp2).logits, dim=-1).detach().cpu().numpy()[0]
        e2 = float(o2[2]) if o2.shape[0] > 2 else float(o2[-1])
        return (e1 + e2) / 2.0
    def score(self, claim, evidence_list):
        scores=[]
        for ev in evidence_list:
            sc = self.entail_score(ev, claim)
            scores.append(sc)
        return float(np.mean(scores)), scores
