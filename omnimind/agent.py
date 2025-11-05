from .retriever import HybridRetriever
from .rag import synthesize_answer
from .tools.tool_registry import list_tools, call_tool

SYSTEM_HINT = """You are OmniMind. When uncertain or when a tool is required, state an intent,
choose a tool with arguments, execute, then continue. Keep answers grounded in retrieved evidence."""

def self_critique(query, answer, ctxs):
    # minimal: check if the answer quotes sources; look for contradictions
    missing_sources = "[SOURCE]" not in answer
    flags = []
    if missing_sources: flags.append("No explicit sources.")
    if not ctxs: flags.append("No evidence found.")
    return "Critique: " + ("; ".join(flags) or "Looks consistent with retrieval.")

class Agent:
    def __init__(self, retriever: HybridRetriever, enable_critique=True, max_iters=6):
        self.retriever = retriever
        self.enable_critique = enable_critique
        self.max_iters = max_iters

    def run(self, query: str):
        # 1) retrieve
        ctxs = self.retriever.retrieve(query)
        # 2) naive tool intent detection (MVP)
        if any(t in query.lower() for t in ["calc","calculate","sqrt","^","sin(","cos("]):
            tool_out = call_tool("calculator", expression=query.split("calc")[-1].strip() or query)
            tool_note = f"\n[TOOL=calculator] {tool_out}\n"
        else:
            tool_note = ""
        # 3) synthesize
        answer = synthesize_answer(query, ctxs) + tool_note
        # 4) self-critique
        if self.enable_critique:
            critique = self_critique(query, answer, ctxs)
            answer = answer + "\n" + critique
        # 5) add minimal citations
        cites = "\nSources:\n" + "\n".join({f"- {c['source']} (score={c['_rank']:.3f})" for c in ctxs})
        return answer + "\n" + cites
