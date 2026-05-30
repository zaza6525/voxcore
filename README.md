# VoxCore

Local voice assistant pipeline — STT → LLM → TTS, 100% offline.

**Stack** : faster-whisper + Qwen3.6 + Voxtral (ou Supertonic/Piper)

**Installation** :
```bash
pip install voxcore
voxcore --config config.yaml
```

**Usage** :
```bash
# Microphone → réponse vocale
voxcore live

# Audio file → texte
voxcore stt input.wav

# Texte → audio
voxcore tts "Hello world" -o output.mp3
```

**Config** : `config.yaml` dans le dossier courant ou `~/.config/voxcore/config.yaml`

## Architecture
- `stt/` — Speech-to-Text (faster-whisper)
- `llm/` — Local LLM (llama.cpp, vLLM, LM Studio)
- `tts/` — Text-to-Speech (Voxtral, Piper, Supertonic)
- `pipeline/` — Orchestration complète

## Requirements
- Python 3.10+
- GPU NVIDIA recommandé (CUDA 12+)
- RAM : 16GB minimum

## License
MIT
