import yaml, sys
from omnimind.memory import VectorMemory
from omnimind.retriever import HybridRetriever
from omnimind.agent import Agent

cfg = yaml.safe_load(open("config.yaml"))
vec = VectorMemory(cfg["paths"]["vector_index"], cfg["paths"]["docstore"], cfg["models"]["embed_text"])
rtv = HybridRetriever(vec, cfg["models"]["cross_encoder"], cfg["retrieval"]["top_k"], cfg["retrieval"]["rerank_k"])
agent = Agent(rtv, enable_critique=cfg["agent"]["self_critique"], max_iters=cfg["agent"]["max_iters"])

q = " ".join(sys.argv[1:]) or "calc 2+2*3"
print(agent.run(q))
