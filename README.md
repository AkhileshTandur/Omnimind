# 🧠 OmniMind — Multimodal RAG + Agentic AI System

**OmniMind** is an open-source, local AI assistant that can **read your data, reason about it, and respond intelligently**.  
It unifies text, image, and audio ingestion with retrieval-augmented generation (RAG), a lightweight knowledge graph, tool use, and self-critique — all exposed through a FastAPI backend and a one-file React web chat UI.


## 🚀 Features

| Capability | Description |
|-------------|-------------|
| 📄 **Text / Image / Audio Ingestion** | Reads `.txt`, `.md`, `.jpg`, `.png`, `.mp3`, `.wav` into embeddings using Sentence-Transformers, CLIP, and Whisper. |
| 🔍 **Vector Memory** | FAISS-based semantic search for relevant document chunks. |
| 🧩 **Knowledge Graph** | Extracts entities and relations (via spaCy) and stores them in a simple graph structure. |
| 💬 **RAG Agent** | Retrieval-augmented generation that synthesizes grounded answers from evidence. |
| 🧮 **Tool Calling** | Extensible tool registry (e.g., built-in calculator). |
| 🧠 **Self-Critique** | Agent reviews its own answers and flags missing evidence. |
| 🌐 **FastAPI Server** | `/ingest`, `/query`, `/agent`, `/tools`, `/health` endpoints. |
| 💻 **React Chat UI** | Clean, responsive front-end built with TailwindCSS + React 18 (CDN-based, no build tools). |
| 🔒 **Runs Locally** | 100% offline — no external APIs required. |


## 🧩 Architecture Overview

                 ┌────────────────────┐
                 │  User / Web UI     │
                 └─────────┬──────────┘
                           │ REST / JSON
                 ┌─────────▼──────────┐
                 │     FastAPI App    │
                 ├────────────────────┤
                 │ /ingest  /query    │
                 │ /agent   /tools    │
                 └─────────┬──────────┘
                           │
         ┌─────────────────▼──────────────────┐
         │         Agent / RAG Core           │
         ├────────────────────────────────────┤
         │ Retriever (FAISS + CrossEncoder)   │
         │ Knowledge Graph (NetworkX)         │
         │ Tools (Calculator, etc.)           │
         │ Self-Critique & Evidence Synthesis │
         └─────────────────┬──────────────────┘
                           │
              ┌────────────▼────────────┐
              │  Vector Store / Memory  │
              │   + Docstore + KG       │
              └─────────────────────────┘
🏗️ Project Structure
omnimind/
├─ omnimind/
│  ├─ memory.py           # Vector memory (FAISS)
│  ├─ ingest_text.py      # Text chunking
│  ├─ ingest_image.py     # CLIP embeddings
│  ├─ ingest_audio.py     # Whisper/Faster-Whisper transcription
│  ├─ kg.py               # Knowledge graph builder
│  ├─ retriever.py        # Hybrid retrieval + re-ranking
│  ├─ rag.py              # Evidence synthesis
│  ├─ agent.py            # Agent loop + tool use + self-critique
│  ├─ tools/              # Tool registry and built-ins
│  ├─ app.py              # FastAPI backend
│  └─ evaluate.py         # Retrieval / RAG evaluation harness
├─ scripts/               # CLI scripts (ingest, query, etc.)
├─ data/raw/              # Input files (.txt/.md/.jpg/.wav)
├─ data/processed/        # Vector index, docstore, KG
└─ web/index.html         # React chat UI

⚙️ Setup
1️⃣ Create and activate a virtual environment
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
# or bash
source .venv/bin/activate

2️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm


(For audio features: pip install faster-whisper ffmpeg-python and ensure ffmpeg is installed.)

🧾 Configuration

Edit config.yaml to adjust:

model names (sentence-transformers/all-MiniLM-L6-v2, cross-encoder/ms-marco-MiniLM-L-6-v2)

chunk size / overlap

top-k retrieval

paths for data and models

▶️ Usage
Ingest data
python scripts/ingest.py

Build knowledge graph
python scripts/build_kg.py

Ask a question (retrieval only)
python scripts/query.py "What is OmniMind?"

Run the full agent
python scripts/run_agent.py "Describe OmniMind."

Start the API server
uvicorn omnimind.app:app --reload


Visit http://127.0.0.1:8000/docs
 for interactive API docs.

💻 Web Chat UI

Serve the front-end:

cd web
python -m http.server 5500


Open http://127.0.0.1:5500

→ Ensure API base URL is http://127.0.0.1:8000
→ Click Ingest data then chat with your AI assistant.

🧮 Example Output
Question: calc 3*(5+2)

Evidence considered:
- OmniMind is a multimodal RAG agent with a vector store and a knowledge graph.

Synthesis:
Based on the retrieved evidence, here is a concise answer:
OmniMind is a multimodal RAG agent with a vector store and a knowledge graph.
[TOOL=calculator] {'result': 21}

Critique: Looks consistent with retrieved evidence.
Sources:
- data/raw/sample.txt (rank=8.128)

🧰 Extending OmniMind
Feature	How to add it
🔧 New Tools	Add functions in omnimind/tools/builtin.py and register in tool_registry.py.
🧠 Better KG	Swap NetworkX for Neo4j and update kg.py.
🗣️ Voice Assistant	Use Whisper for STT + pyttsx3 for TTS.
🌍 Cloud Deployment	Containerize with Docker + run behind Nginx or Render.
🤖 Bigger Models	Change model names in config.yaml (e.g., BAAI/bge-small-en).
📊 Evaluation

Create data/processed/eval_qa.jsonl:

{"query":"What is OmniMind?","answers":["OmniMind is a multimodal RAG agent"],"positive_ids":["<doc_id>"]}


Run:

python -m omnimind.evaluate --eval_jsonl data/processed/eval_qa.jsonl

🧭 Roadmap

 Add LLM-based answer synthesis (Phi-3, Mistral, etc.)

 Neo4j-powered Knowledge Graph

 Web Search Tool plugin

 Voice interface

 Docker + Hugging Face Space deployment

📜 License

MIT License © 2025

🤝 Credits

Built with ❤️ using:

PyTorch

Sentence-Transformers

FAISS

spaCy

FastAPI

React

TailwindCSS

✨ Author’s Note

OmniMind was created to help you understand how AI systems actually work under the hood — not just call an API.
It’s a full end-to-end architecture: ingestion → memory → reasoning → tools → reflection → interface.
Use it as your personal research assistant, or a foundation to build your own custom copilots.

## 📘 To publish

omnimind/
├─ omnimind/
├─ scripts/
├─ web/
├─ data/
├─ config.yaml
├─ requirements.txt
└─ README.md

2. Initialize Git & push to GitHub:
```bash
git init
git add .
git commit -m "Initial commit: OmniMind RAG agent"
git branch -M main
git remote add origin https://github.com/<yourusername>/OmniMind.git
git push -u origin main

