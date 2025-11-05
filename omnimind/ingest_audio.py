from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .utils import sha1
from .ingest_text import simple_chunks

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

class AudioTranscriber:
    """
    Minimal wrapper that prefers faster-whisper (GPU/CPU), falls back to openai-whisper.
    Usage:
        tr = AudioTranscriber(model_size="base")
        text, segments = tr.transcribe("/path/audio.mp3")
    """
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        self.backend = None
        self.model = None
        self.model_size = model_size
        self.device = device  # "cuda", "cpu", or None to auto

        # Try faster-whisper first (fast, lighter)
        try:
            from faster_whisper import WhisperModel  # type: ignore
            compute_type = "float16" if (self._has_cuda() and (device in (None, "cuda"))) else "int8"
            self.model = WhisperModel(model_size, device=self._pick_device(), compute_type=compute_type)
            self.backend = "faster-whisper"
            return
        except Exception:
            pass

        # Fallback: openai-whisper (reference implementation)
        try:
            import whisper  # type: ignore
            self.model = whisper.load_model(model_size, device=self._pick_device())
            self.backend = "whisper"
            return
        except Exception as e:
            raise RuntimeError(
                "No ASR backend available. Install either:\n"
                "  pip install faster-whisper\n"
                "or\n"
                "  pip install openai-whisper\n"
                f"Original error: {type(e).__name__}: {e}"
            )

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _pick_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if self._has_cuda() else "cpu"

    def transcribe(self, path: str) -> Tuple[str, List[Dict[str, float]]]:
        """
        Returns:
            transcript_text, segments (each has start, end, text if available)
        """
        if self.backend == "faster-whisper":
            segments, info = self.model.transcribe(path, beam_size=1)
            segs = []
            full = []
            for s in segments:
                segs.append({"start": float(s.start), "end": float(s.end), "text": s.text})
                full.append(s.text.strip())
            return " ".join(full).strip(), segs

        # openai-whisper
        if self.backend == "whisper":
            import whisper
            out = self.model.transcribe(path)  # type: ignore[attr-defined]
            text = out.get("text", "").strip()
            segs = [
                {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": s.get("text", "")}
                for s in out.get("segments", [])
            ]
            return text, segs

        raise RuntimeError("No transcription backend initialized.")

def prepare_audio_docs(raw_dir: str, model_size: str = "base", device: Optional[str] = None,
                       chunk_size: int = 800, chunk_overlap: int = 120) -> List[Dict[str, Any]]:
    """
    Walks data/raw for audio files, transcribes, chunks, and returns doc dicts
    compatible with VectorMemory.add_texts().
    """
    transcriber = AudioTranscriber(model_size=model_size, device=device)
    docs: List[Dict[str, Any]] = []
    for p in Path(raw_dir).rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS:
            try:
                transcript, segments = transcriber.transcribe(str(p))
                if not transcript:
                    continue
                for i, chunk in enumerate(simple_chunks(transcript, size=chunk_size, overlap=chunk_overlap)):
                    docs.append({
                        "id": sha1(f"{p}:{i}:audio"),
                        "source": str(p),
                        "type": "audio",
                        "text": chunk,
                        "meta": {
                            "backend": transcriber.backend,
                            "model_size": model_size,
                            # keep a small sample of segments for traceability (optional)
                            "segments_sample": segments[:5] if segments else [],
                        }
                    })
            except Exception as e:
                # Non-fatal: skip file but print a helpful message
                print(f"[ingest_audio] Failed {p.name}: {type(e).__name__}: {e}")
    return docs
