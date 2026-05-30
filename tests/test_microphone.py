"""Tests microphone + VAD (numpy only — no sounddevice needed)."""
import pytest
import numpy as np
from voxcore_local.microphone import VAD


def test_vad_silence():
    """Le silence seul ne déclenche rien."""
    vad = VAD(threshold=0.01, min_silence_sec=0.5, sample_rate=16000, chunk_size=1600)
    silence = np.zeros(1600, dtype=np.float32)
    
    for _ in range(10):
        result = vad.process_chunk(silence)
        assert result is None


def test_vad_speech_then_silence():
    """Parole → silence ≥ min_silence → phrase détectée."""
    vad = VAD(threshold=0.01, min_silence_sec=0.1, sample_rate=16000, chunk_size=1600)
    
    speech = np.random.randn(1600).astype(np.float32) * 0.1
    assert vad.process_chunk(speech) is None  # Parle, pas de trigger
    
    silence = np.zeros(1600, dtype=np.float32)
    # 1600 samples = 0.1s = min_silence_sec → trigger
    result = vad.process_chunk(silence)
    assert result is not None
    assert len(result) > 0  # WAV non vide


def test_vad_force_end():
    """force_end retourne le buffer restant."""
    vad = VAD(threshold=0.01)
    speech = np.random.randn(1600).astype(np.float32) * 0.1
    vad.process_chunk(speech)
    
    result = vad.force_end()
    assert result is not None


def test_vad_short_silence_no_trigger():
    """Petit silence dans la parole ne déclenche pas."""
    vad = VAD(threshold=0.01, min_silence_sec=1.0, sample_rate=16000, chunk_size=1600)
    
    speech = np.random.randn(1600).astype(np.float32) * 0.1
    silence = np.zeros(1600, dtype=np.float32)
    
    vad.process_chunk(speech)
    vad.process_chunk(silence)  # 0.1s de silence < 1s → pas de trigger
    result = vad.process_chunk(speech)  # Reprendre à parler
    
    assert result is None


def test_vad_output_is_wav():
    """Le résultat du VAD est un fichier WAV valide (en-tête RIFF)."""
    vad = VAD(threshold=0.01, min_silence_sec=0.1, sample_rate=16000, chunk_size=1600)
    speech = np.random.randn(1600).astype(np.float32) * 0.1
    vad.process_chunk(speech)
    
    silence = np.zeros(1600, dtype=np.float32)
    result = vad.process_chunk(silence)
    
    assert result is not None
    assert result.startswith(b'RIFF')
