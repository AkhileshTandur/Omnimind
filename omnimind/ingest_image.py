from PIL import Image
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
from .utils import sha1

class ImageEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.proc = CLIPProcessor.from_pretrained(model_name)

    def encode(self, images):
        inputs = self.proc(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embs = self.model.get_image_features(**inputs)
            embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        return embs.cpu().numpy()

def prepare_image_docs(raw_dir):
    paths = [p for p in Path(raw_dir).rglob("*") if p.suffix.lower() in {".png",".jpg",".jpeg"}]
    docs = []
    for p in paths:
        docs.append({
            "id": sha1(str(p)),
            "source": str(p),
            "type": "image",
            "text": f"[IMAGE] {p.name} (visual semantics via CLIP)"  # textual proxy for indexing
        })
    return docs
