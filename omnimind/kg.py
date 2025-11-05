import spacy, networkx as nx
from pathlib import Path
from .utils import load_jsonl
import pickle

# english core
_nlp = None
def nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def extract_triples(text):
    doc = nlp()(text)
    ents = [(e.text, e.label_) for e in doc.ents]
    triples = []
    # very simple SVO-style pattern (MVP); refine later
    for sent in doc.sents:
        subj, verb, obj = None, None, None
        for token in sent:
            if token.dep_ in {"nsubj","nsubjpass"}: subj = token.text
            if token.pos_ == "VERB": verb = token.lemma_
            if token.dep_ in {"dobj","pobj"}: obj = token.text
        if subj and verb and obj:
            triples.append((subj, verb, obj))
    return ents, triples



def build_graph(docstore_path, out_path):
    G = nx.MultiDiGraph()
    texts = [d for d in load_jsonl(docstore_path) if d.get("type") == "text"]
    for d in texts:
        ents, triples = extract_triples(d["text"])
        for e, label in ents:
            G.add_node(e, label=label)
        for s, v, o in triples:
            G.add_edge(s, o, rel=v, source=d["source"], doc_id=d["id"])
    
    # --- Save graph safely (NetworkX 3.x fix) ---
    with open(out_path, "wb") as f:
        pickle.dump(G, f)
    print(f"Knowledge graph saved to {out_path}")
    return G

