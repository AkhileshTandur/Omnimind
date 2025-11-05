import yaml, sys
from omnimind.memory import VectorMemory
from omnimind.retriever import HybridRetriever
from omnimind.rag import synthesize_answer

cfg = yaml.safe_load(open("config.yaml"))
vec = VectorMemory(cfg["paths"]["vector_index"], cfg["paths"]["docstore"], cfg["models"]["embed_text"])
rtv = HybridRetriever(vec, cfg["models"]["cross_encoder"], cfg["retrieval"]["top_k"], cfg["retrieval"]["rerank_k"])

q = " ".join(sys.argv[1:]) or "What do these documents say about X?"
ctxs = rtv.retrieve(q)
print(synthesize_answer(q, ctxs))
