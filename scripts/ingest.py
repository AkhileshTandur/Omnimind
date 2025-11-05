import yaml, json
from pathlib import Path
from omnimind.memory import VectorMemory
from omnimind.ingest_text import prepare_text_docs
from omnimind.ingest_image import prepare_image_docs
# from omnimind.ingest_audio import prepare_audio_docs  # optional

cfg = yaml.safe_load(open("config.yaml"))
vec = VectorMemory(cfg["paths"]["vector_index"], cfg["paths"]["docstore"], cfg["models"]["embed_text"])

docs = []
docs += prepare_text_docs(cfg["paths"]["data_raw"])
docs += prepare_image_docs(cfg["paths"]["data_raw"])
# docs += prepare_audio_docs(cfg["paths"]["data_raw"], model_size="base")  # optional

print(f"Ingesting {len(docs)} chunks...")
if not docs:                     # ← add this guard
    print("No files found under data/raw. Add .txt/.md/.jpg/.png (and audio if enabled).")
else:
    vec.add_texts(docs)
    print("Done.")
