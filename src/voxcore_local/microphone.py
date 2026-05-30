"""Microphone capture avec Voice Activity Detection (VAD)."""
import io
import queue
import struct
import threading
import wave
from typing import Callable, Optional

import numpy as np


class VAD:
    """Simple energy-based VAD — pas de dépendance externe."""
    
    def __init__(self, threshold: float = 0.01, min_silence_sec: float = 0.8,
                 sample_rate: int = 16000, chunk_size: int = 1600):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_silence_samples = int(min_silence_sec * sample_rate)
        
        self._buffer = bytearray()
        self._silence_samples = 0
        self._has_speech = False
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[bytes]:
        """Process a chunk of audio. Returns WAV bytes on phrase end, None otherwise."""
        rms = np.sqrt(np.mean(audio_chunk.astype(np.float64) ** 2))
        is_speech = rms > self.threshold
        
        if is_speech:
            self._buffer.extend(audio_chunk.astype(np.int16).tobytes())
            self._silence_samples = 0
            self._has_speech = True
            return None
        else:
            if self._has_speech:
                self._silence_samples += len(audio_chunk)
                if self._silence_samples >= self.min_silence_samples:
                    wav = self._pcm_to_wav(bytes(self._buffer))
                    self._buffer = bytearray()
                    self._silence_samples = 0
                    self._has_speech = False
                    return wav
            return None
    
    def force_end(self) -> Optional[bytes]:
        if self._buffer:
            wav = self._pcm_to_wav(bytes(self._buffer))
            self._buffer = bytearray()
            self._has_speech = False
            return wav
        return None
    
    def _pcm_to_wav(self, pcm: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm)
        return buf.getvalue()


class MicrophoneCapture:
    """Capture microphone continue avec VAD.
    
    Imports sounddevice only when needed.
    """
    
    def __init__(self, device_index: Optional[int] = None,
                 vad_threshold: float = 0.01,
                 min_silence: float = 0.8,
                 sample_rate: int = 16000,
                 channels: int = 1):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = 1600
        self.vad = VAD(
            threshold=vad_threshold,
            min_silence_sec=min_silence,
            sample_rate=sample_rate,
            chunk_size=self.chunk_size,
        )
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[bytes], None]] = None
    
    @staticmethod
    def list_devices():
        """List audio devices. Returns empty list if sounddevice unavailable."""
        try:
            import sounddevice as sd
            return sd.query_devices()
        except Exception:
            return []
    
    def start(self, callback: Callable[[bytes], None]):
        """Start capturing from microphone."""
        import sounddevice as sd  # lazy
        
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
    
    def _capture_loop(self):
        import sounddevice as sd
        
        def callback(indata, frames, time, status):
            if status:
                return
            if self._running:
                audio = indata.copy().ravel()
                result = self.vad.process_chunk(audio)
                if result and self._callback:
                    self._callback(result)
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.chunk_size,
            device=self.device_index,
            callback=callback,
        ):
            while self._running:
                import time
                time.sleep(0.1)
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
