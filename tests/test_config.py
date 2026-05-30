"""Tests de base VoxCore."""
import pytest
from voxcore_local.config import load_config, DEFAULT_CONFIG
from voxcore_local.tts import sanitize, split_text


def test_default_config():
    cfg = load_config("/nonexistent/path.yaml")
    assert "stt" in cfg
    assert "llm" in cfg
    assert "tts" in cfg
    assert cfg["stt"]["language"] == "fr"


def test_sanitize_emoji():
    assert sanitize("Hello 👋 world!") == "Hello world!"
    assert sanitize("Pas de problème ✅") == "Pas de problème"


def test_split_text():
    text = "Première phrase. Deuxième phrase. Troisième phrase finale !"
    chunks = split_text(text, 20)  # max_chars=20 → forcé à 80 par le max()
    # Avec max_chars=80, tout tient dans un seul chunk (59 chars)
    assert len(chunks) == 1
    assert chunks[0] == text
    
    # Test avec texte long qui force le split
    long_text = " ".join([f"Phrase {i}." for i in range(20)])
    chunks2 = split_text(long_text, 80)
    assert len(chunks2) >= 2
    for chunk in chunks2:
        assert len(chunk) <= 80


def test_split_text_long_word():
    text = "C'estunmottrèslongquisansespace"
    chunks = split_text(text, 20)
    assert len(chunks) >= 1
    assert " ".join(chunks) == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
