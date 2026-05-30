"""Configuration YAML."""
import os
from pathlib import Path
from typing import Optional
import yaml

DEFAULT_CONFIG = {
    "stt": {
        "model_size": "large-v3",
        "device": "cuda",
        "language": "fr",
    },
    "llm": {
        "base_url": "http://127.0.0.1:1234/v1",
        "model": "local",
        "api_key": "sk-local",
        "system_prompt": "Tu es un assistant vocal utile. Réponds de manière concise et naturelle, comme si tu parlais à quelqu'un.",
        "max_tokens": 512,
        "temperature": 0.7,
    },
    "tts": {
        "backend": "voxtral",  # ou "piper"
        "url": "http://192.168.1.90:9000/tts",
        "voice": "fr_female",
        "speed": 1.0,
        "chunk_chars": 320,
    },
    "output": {
        "format": "mp3",
        "dir": "/tmp/voxcore",
    },
    "memory": {
        "short_term_turns": 20,
        "filter_threshold": 0.3,
    },
}


def load_config(path: Optional[str] = None) -> dict:
    """Charge la config YAML."""
    if path:
        config_path = Path(path)
        if not config_path.exists():
            return DEFAULT_CONFIG.copy()
    else:
        # Cherche dans l'ordre : argument, CWD, ~/.config/voxcore/
        for p in [Path.cwd() / "config.yaml", Path.home() / ".config" / "voxcore" / "config.yaml"]:
            if p.exists():
                config_path = p
                break
        else:
            return DEFAULT_CONFIG.copy()
    
    with open(config_path) as f:
        user_config = yaml.safe_load(f) or {}
    
    # Merge profond
    config = DEFAULT_CONFIG.copy()
    for section in config:
        if section in user_config:
            if isinstance(config[section], dict):
                config[section].update(user_config[section])
            else:
                config[section] = user_config[section]
    
    return config
