from sentence_transformers import CrossEncoder
from .memory import VectorMemory

class HybridRetriever:
    def __init__(self, vecmem: VectorMemory, cross_encoder_name: str, top_k=12, rerank_k=6):
        self.vecmem = vecmem
        self.rank = CrossEncoder(cross_encoder_name)
        self.top_k, self.rerank_k = top_k, rerank_k

    def retrieve(self, query: str):
        initial = self.vecmem.search(query, k=self.top_k)
        pairs = [(query, d["text"]) for d in initial]
        scores = self.rank.predict(pairs)
        rescored = sorted(
            [dict(d, _rank=float(s)) for d, s in zip(initial, scores)],
            key=lambda x: x["_rank"], reverse=True
        )[:self.rerank_k]
        return rescored
