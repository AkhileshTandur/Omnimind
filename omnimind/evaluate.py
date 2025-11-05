from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml
import math
from collections import defaultdict, Counter

from .memory import VectorMemory
from .retriever import HybridRetriever

"""
Lightweight evaluation harness for:
1) Retrieval: Recall@K and MRR@K against gold supporting IDs.
2) RAG Answer Quality: token-overlap Precision/Recall/F1 vs. gold answers (bag-of-words).

Expected evaluation file format (JSONL):
{
  "query": "What is X?",
  "answers": ["Gold canonical answer."],                # one or more acceptable refs
  "positive_ids": ["<doc_id_1>", "<doc_id_2>", ...]     # doc IDs considered supporting
}

You can build this file gradually as you use the system.
"""

def tokenize(s: str) -> List[str]:
    return [t for t in "".join([c.lower() if c.isalnum() else " " for c in s]).split() if t]

def f1(pred: str, golds: List[str]) -> Tuple[float, float, float]:
    """
    Token-overlap F1 against the best matching gold.
    Returns (precision, recall, f1).
    """
    p_tokens = Counter(tokenize(pred))
    best = (0.0, 0.0, 0.0)
    for g in golds:
        g_tokens = Counter(tokenize(g))
        intersection = sum((p_tokens & g_tokens).values())
        p_total = sum(p_tokens.values()) or 1
        g_total = sum(g_tokens.values()) or 1
        prec = intersection / p_total
        rec = intersection / g_total
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if f1 > best[2]:
            best = (prec, rec, f1)
    return best

def recall_at_k(retrieved_ids: List[str], positive_ids: List[str], k: int) -> float:
    top = set(retrieved_ids[:k])
    pos = set(positive_ids)
    if not pos:
        return 0.0
    return 1.0 if top & pos else 0.0

def mrr_at_k(retrieved_ids: List[str], positive_ids: List[str], k: int) -> float:
    pos = set(positive_ids)
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in pos:
            return 1.0 / i
    return 0.0

def evaluate_file(eval_path: str, retriever: HybridRetriever, ks=(1, 3, 5, 10)) -> Dict[str, Any]:
    results = {
        "retrieval": {f"Recall@{k}": [] for k in ks} | {f"MRR@{k}": [] for k in ks},
        "rag": {"P": [], "R": [], "F1": []},
        "count": 0,
    }

    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            query = ex["query"]
            gold_answers = ex.get("answers", [])
            pos_ids = ex.get("positive_ids", [])

            ctxs = retriever.retrieve(query)
            retrieved_ids = [c["id"] for c in ctxs]

            # Retrieval metrics
            for k in ks:
                results["retrieval"][f"Recall@{k}"].append(recall_at_k(retrieved_ids, pos_ids, k))
                results["retrieval"][f"MRR@{k}"].append(mrr_at_k(retrieved_ids, pos_ids, k))

            # RAG lexical F1 against top-3 concatenation (cheap proxy)
            predicted = " ".join([c["text"] for c in ctxs[:3]])
            P, R, F = f1(predicted, gold_answers) if gold_answers else (0.0, 0.0, 0.0)
            results["rag"]["P"].append(P); results["rag"]["R"].append(R); results["rag"]["F1"].append(F)

            results["count"] += 1

    # Aggregate
    agg = {
        "retrieval": {k: (sum(v) / len(v) if v else 0.0) for k, v in results["retrieval"].items()},
        "rag": {k: (sum(v) / len(v) if v else 0.0) for k, v in results["rag"].items()},
        "count": results["count"],
    }
    return agg

def main():
    parser = argparse.ArgumentParser(description="Evaluate OmniMind retrieval and RAG.")
    parser.add_argument("--eval_jsonl", type=str, default="data/processed/eval_qa.jsonl",
                        help="Path to JSONL eval file.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--top_k", type=int, default=None, help="Override top_k for retrieval.")
    parser.add_argument("--rerank_k", type=int, default=None, help="Override rerank_k for retrieval.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    vec = VectorMemory(cfg["paths"]["vector_index"], cfg["paths"]["docstore"], cfg["models"]["embed_text"])
    top_k = args.top_k or cfg["retrieval"]["top_k"]
    rerank_k = args.rerank_k or cfg["retrieval"]["rerank_k"]
    rtv = HybridRetriever(vec, cfg["models"]["cross_encoder"], top_k, rerank_k)

    report = evaluate_file(args.eval_jsonl, rtv)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
