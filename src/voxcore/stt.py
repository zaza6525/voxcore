"""STT — Speech-to-Text via faster-whisper."""
import os
from typing import Optional
from faster_whisper import WhisperModel

class STTEngine:
    """faster-whisper STT avec chargement lazy."""
    
    def __init__(self, model_size: str = "large-v3", device: str = "auto", language: str = "fr"):
        self.model_size = model_size
        self.device = device
        self.language = language
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="int8" if self.device == "cpu" else "float16",
                )
            except RuntimeError:
                # Fallback CPU si CUDA OOM (LLM prend déjà la VRAM)
                print(f"⚠️  STT: {self.device} indisponible, fallback CPU int8")
                self._model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                )
        return self._model
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe un fichier audio en texte."""
        lang = language or self.language
        segments, info = self.model.transcribe(audio_path, language=lang, beam_size=5)
        text = " ".join(seg.text for seg in segments)
        return text.strip()
    
    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000, language: Optional[str] = None) -> str:
        """Transcribe des bytes audio (WAV 16kHz mono attendu)."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            f.flush()
            try:
                return self.transcribe(f.name, language)
            finally:
                os.unlink(f.name)
