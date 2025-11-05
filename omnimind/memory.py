import faiss, json, os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .utils import ensure_dir, append_jsonl

class VectorMemory:
    def __init__(self, index_path, docstore_path, model_name):
        self.index_path = index_path
        self.docstore_path = docstore_path
        ensure_dir(Path(index_path).parent)
        ensure_dir(Path(docstore_path).parent)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.ids = []  # match ordering with FAISS
        self.meta = [] # doc metadata

        if Path(index_path).exists() and Path(docstore_path).exists():
            self._load()

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.docstore_path, "r", encoding="utf-8") as f:
            self.meta = [json.loads(line) for line in f if line.strip()]
        self.ids = [m["id"] for m in self.meta]

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.docstore_path, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def add_texts(self, docs):
        texts = [d["text"] for d in docs]
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embs = np.array(embs).astype("float32")
        self.index.add(embs)
        for d in docs:
            self.meta.append(d)
            self.ids.append(d["id"])
        self._save()

    def search(self, query, k=10):
        q = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(q).astype("float32"), k)
        out = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1: continue
            m = self.meta[idx]
            m = dict(m)  # copy
            m["_score"] = float(dist)
            out.append(m)
        return out
