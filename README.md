# 🎙️ VoxCore

**The Ultimate Local Voice Intelligence Pipeline**

VoxCore is a high-performance, 100% offline voice assistant pipeline designed for privacy, speed, and complete local control. It orchestrates the entire journey from your voice to a spoken response without a single byte leaving your machine.

![Architecture](https://img.shields.io/badge/Architecture-STT%20%E2%86%92%20LLM%20%E2%86%92%20TTS-blueviolet)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Overview

VoxCore isn't just another wrapper; it's a professional-grade pipeline optimized for NVIDIA GPUs and local LLMs. It solves the "latency problem" of local AI by implementing an intelligent routing system and a semantic memory layer.

### 🛠 The Tech Stack
- **STT (Speech-to-Text)**: `faster-whisper` (industry standard for speed and accuracy).
- **Brain (LLM)**: OpenAI-compatible local endpoints (LM Studio, vLLM, llama.cpp) — optimized for **Qwen 3.6**.
- **TTS (Text-to-Speech)**: Hybrid routing between **Voxtral** (High fidelity/PC2) and **Piper** (Ultra-fast local fallback).

## 🏗 Architecture

```mermaid
graph LR
    A[Microphone] --> B[VAD / Streaming STT]
    B --> C[Semantic Memory Filter]
    C --> D[Local LLM]
    D --> E[TTS Router]
    E --> F[Voxtral / Piper]
    F --> G[Speaker]
```

## 📦 Installation

### Prerequisites
- **Python 3.10+**
- **NVIDIA GPU** (CUDA 12+)
- **RAM**: 16GB minimum

### Quick Start
```bash
# Install the package
pip install voxcore

# Run the live interactive session
voxcore live --config config.yaml
```

## 🛠 Usage

### 🎤 Live Interaction
Start a real-time conversation where the AI listens, remembers, and speaks back:
```bash
voxcore live
```

### 📝 Transcription (STT)
Convert any audio file to text with high precision:
```bash
voxcore stt input.wav
```

### 🗣 Synthesis (TTS)
Generate high-quality speech from text:
```bash
voxcore tts "Bonjour Ali, le système est opérationnel." -o output.mp3
```

## ⚙️ Configuration

Configure your endpoints and voices in `~/.config/voxcore/config.yaml`:

```yaml
llm:
  endpoint: "http://localhost:1234/v1"
  model: "qwen3.6-27b"

tts:
  url: "http://192.168.1.90:9000/tts"
  voice: "fr_female"
  fallback: "piper"
```

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
