"""Lecteur audio pour la sortie vocale.

Imports sounddevice/soundfile lazily.
"""
import io
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np


class AudioPlayer:
    """Lecture audio non-bloquante.
    
    Importe sounddevice uniquement à l'usage.
    """
    
    def __init__(self, device_index: Optional[int] = None):
        self.device_index = device_index
        self._playing = False
    
    def play_bytes(self, audio_bytes: bytes, format: str = "WAV"):
        """Play raw audio bytes in background thread."""
        self._stop()
        
        t = threading.Thread(
            target=self._play_thread,
            args=(audio_bytes, format),
            daemon=True,
        )
        t.start()
    
    def play_file(self, path: str):
        """Play an audio file in background thread."""
        data = Path(path).read_bytes()
        self.play_bytes(data)
    
    def _play_thread(self, audio_bytes: bytes, fmt: str):
        import sounddevice as sd
        import soundfile as sf
        
        buf = io.BytesIO(audio_bytes)
        data, sr = sf.read(buf, dtype=np.float32)
        
        self._playing = True
        try:
            sd.play(data, sr, device=self.device_index)
            sd.wait()
        except Exception:
            pass
        finally:
            self._playing = False
    
    def _stop(self):
        if self._playing:
            try:
                import sounddevice as sd
                sd.stop()
            except Exception:
                pass
        self._playing = False
    
    @property
    def is_playing(self) -> bool:
        return self._playing
