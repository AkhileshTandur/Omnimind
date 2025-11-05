import yaml
from omnimind.kg import build_graph
cfg = yaml.safe_load(open("config.yaml"))
G = build_graph(cfg["paths"]["docstore"], cfg["paths"]["kg_graph"])
print(f"KG nodes={len(G.nodes)}, edges={len(G.edges)}")
