from __future__ import annotations
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .memory import VectorMemory
from .ingest_text import prepare_text_docs
from .ingest_image import prepare_image_docs
from .retriever import HybridRetriever
from .agent import Agent

# ---------- Load config ----------
CFG = yaml.safe_load(open("config.yaml", "r"))

# ---------- Singletons ----------
_vec: Optional[VectorMemory] = None
_rtv: Optional[HybridRetriever] = None
_agent: Optional[Agent] = None

def _ensure_components():
    global _vec, _rtv, _agent
    if _vec is None:
        _vec = VectorMemory(
            CFG["paths"]["vector_index"],
            CFG["paths"]["docstore"],
            CFG["models"]["embed_text"],
        )
    if _rtv is None:
        _rtv = HybridRetriever(
            _vec,
            CFG["models"]["cross_encoder"],
            CFG["retrieval"]["top_k"],
            CFG["retrieval"]["rerank_k"],
        )
    if _agent is None:
        _agent = Agent(
            _rtv,
            enable_critique=CFG["agent"]["self_critique"],
            max_iters=CFG["agent"]["max_iters"],
        )

# ---------- FastAPI ----------
app = FastAPI(title="OmniMind API", version="0.1.0", description="RAG + Tools + Self-critique")

# (Optional) loosen CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class IngestResponse(BaseModel):
    chunks_added: int = Field(..., description="Number of chunks added to the vector store.")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    contexts: List[Dict[str, Any]]

class AgentRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    answer: str

class ToolsResponse(BaseModel):
    tools: Dict[str, Dict[str, Any]]

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    Ingest files under data/raw:
      - .txt, .md (chunked + embedded)
      - .png, .jpg, .jpeg (image proxies for MVP)
    """
    _ensure_components()
    raw_dir = CFG["paths"]["data_raw"]
    if not Path(raw_dir).exists():
        raise HTTPException(400, f"Raw data directory not found: {raw_dir}")

    text_docs = prepare_text_docs(raw_dir)
    img_docs = prepare_image_docs(raw_dir)
    docs = text_docs + img_docs

    if not docs:
        return IngestResponse(chunks_added=0)

    _vec.add_texts(docs)
    return IngestResponse(chunks_added=len(docs))

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Retrieval + synthesis only (no tool calls or critique).
    Useful for debugging retrieval quality.
    """
    _ensure_components()
    ctxs = _rtv.retrieve(req.query)
    answer = f"(RAG) {req.query}\n\n" + "\n".join([c["text"][:280].replace("\n", " ") for c in ctxs])
    return QueryResponse(answer=answer, contexts=ctxs)

@app.post("/agent", response_model=AgentResponse)
def agent(req: AgentRequest):
    """
    Full agent loop: retrieve → (maybe) tool → synthesize → critique → cite.
    """
    _ensure_components()
    out = _agent.run(req.query)
    return AgentResponse(answer=out)

@app.get("/tools", response_model=ToolsResponse)
def tools():
    _ensure_components()
    return ToolsResponse(tools=_agent.available_tools())
