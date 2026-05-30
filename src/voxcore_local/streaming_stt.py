"""Streaming STT — transcription incrémentale avec faster-whisper.

Permet de transcrire des flux audio en temps réel sans attendre
la fin de l'enregistrement. Utilise le modèle whisper de manière
incrémentale avec beam search progressif.
"""
import io
import os
import tempfile
import time
from typing import Callable, Optional

import numpy as np
from faster_whisper import WhisperModel


class StreamingTranscriber:
    """Transcription streaming avec faster-whisper.
    
    Accumule des chunks audio et produit des transcriptions
    incrémentales à intervalle régulier.
    
    Usage:
        st = StreamingTranscriber(model_size='medium')
        st.start()
        for chunk in audio_stream:
            st.feed_chunk(chunk, sample_rate=16000)
        st.stop()
    """
    
    def __init__(self, model_size: str = "large-v3", 
                 device: str = "cuda",
                 language: str = "fr",
                 interval: float = 1.0,
                 on_partial: Optional[Callable[[str], None]] = None):
        self.model_size = model_size
        self.device = device
        self.language = language
        self.interval = interval
        self.on_partial = on_partial
        
        self._model = None
        self._buffer = bytearray()
        self._sample_rate = 16000
        self._running = False
        self._last_text = ""
    
    @property
    def model(self):
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device="cuda",
                compute_type="float16",
            )
        return self._model
    
    def start(self):
        self._running = True
        self._buffer = bytearray()
        self._last_text = ""
    
    def feed_chunk(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """Ajoute un chunk audio au buffer."""
        self._sample_rate = sample_rate
        self._buffer.extend(audio_data.tobytes())
    
    def transcribe_buffer(self) -> str:
        """Transcrit le buffer actuel (à appeler périodiquement)."""
        if not self._buffer:
            return ""
        
        wav_bytes = self._buffer_to_wav()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(wav_bytes)
            path = f.name
        
        try:
            segments, _ = self.model.transcribe(
                path,
                language=self.language,
                beam_size=3,
                vad_filter=False,
            )
            text = " ".join(seg.text for seg in segments)
            text = text.strip()
            
            # Callback partiel
            if text and text != self._last_text and self.on_partial:
                self.on_partial(text)
            
            self._last_text = text
            return text
        finally:
            os.unlink(path)
    
    def flush(self) -> str:
        """Retourne la transcription complète et vide le buffer."""
        text = self.transcribe_buffer()
        self._buffer = bytearray()
        self._last_text = ""
        return text
    
    def stop(self):
        """Arrête et retourne la transcription finale."""
        self._running = False
        return self.flush()
    
    def _buffer_to_wav(self) -> bytes:
        """Convertit le buffer PCM en WAV."""
        import wave
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self._sample_rate)
            wav.writeframes(bytes(self._buffer))
        return buf.getvalue()
