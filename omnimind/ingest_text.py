import re, os
from pathlib import Path
from .utils import sha1
def simple_chunks(text, size=800, overlap=120):
    tokens = re.split(r'(\s+)', text)
    buf, out, n = [], [], 0
    for t in tokens:
        buf.append(t)
        n += len(t)
        if n >= size:
            out.append("".join(buf))
            buf = buf[-overlap//2:]
            n = sum(len(x) for x in buf)
    if buf: out.append("".join(buf))
    return out

def prepare_text_docs(raw_dir):
    docs = []
    for p in Path(raw_dir).rglob("*"):
        if p.suffix.lower() in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            for i, chunk in enumerate(simple_chunks(text)):
                docs.append({
                    "id": sha1(f"{p}:{i}"),
                    "source": str(p),
                    "type": "text",
                    "text": chunk
                })
    return docs
