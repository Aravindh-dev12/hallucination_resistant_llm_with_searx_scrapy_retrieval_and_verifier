import re
def extract_citations(answer, evidence_list):
    hits=[]
    ans = answer.lower()
    for i, ev in enumerate(evidence_list):
        ev_lower = ev.lower()
        for sent in ev_lower.split('.'):
            s = sent.strip()
            if not s or len(s) < 20: continue
            if s[:20] in ans or s in ans:
                hits.append({'idx':i, 'snippet': ev[:400]})
                break
    return hits
