"""Tests pipeline + config + TTS router."""
import pytest
from voxcore_local.config import load_config, DEFAULT_CONFIG
from voxcore_local.tts import sanitize, split_text, VoxtralTTS


def test_config_defaults():
    cfg = load_config("/nonexistent")
    assert cfg["stt"]["language"] == "fr"
    assert cfg["stt"]["model_size"] == "large-v3"
    assert "llm" in cfg
    assert "tts" in cfg


def test_config_llm_defaults():
    cfg = load_config("/nonexistent")
    assert "1234" in cfg["llm"]["base_url"]
    assert cfg["llm"]["max_tokens"] == 512


def test_sanitize_clean():
    assert sanitize("Hello world!") == "Hello world!"
    assert sanitize("  Espaces   multiples  ") == "Espaces multiples"


def test_sanitize_emoji():
    assert sanitize("Bonjour 👋") == "Bonjour"
    assert sanitize("Test ✅ fait") == "Test fait"


def test_split_text_short():
    text = "Phrase courte."
    chunks = split_text(text, 80)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_long():
    long_text = " ".join([f"Phrase numéro {i}." for i in range(30)])
    chunks = split_text(long_text, 80)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 80


def test_split_text_min_chars():
    """max_chars < 80 est forcé à 80. Le split respecte les phrases."""
    # Sans ponctuation, ça reste un seul chunk (pas de frontière de phrase)
    chunks = split_text("A" * 100, 10)
    assert len(chunks) == 1  # Pas de .!?;: → pas de split possible
    
    # Avec phrases courtes, ça merge dans des chunks ≤ 80
    text = "A. B. C. D. E. F. G. H. I. J."
    chunks = split_text(text, 5)  # forcé à 80 → tout tient
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c) <= 80


def test_sanitize_empty():
    assert sanitize("  👋  ") == ""
    assert sanitize("") == ""
