"""TTS — Text-to-Speech with multiple backends."""
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
from pathlib import Path
from typing import Optional

import re as regex

# Regex pour strip les emojis qui plantent certains TTS
_EMOJI_STRIP = re.compile(
    "["
    "\U0001F000-\U0001FFFF"
    "\u2600-\u27BF"
    "\uFE00-\uFE0F"
    "\u200B-\u200F"
    "\u202A-\u202E"
    "\u2060-\u206F"
    "\uFEFF"
    "]+"
)


def sanitize(text: str) -> str:
    """Nettoie le texte pour le TTS."""
    text = _EMOJI_STRIP.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text en chunks respectant les frontières de phrases."""
    max_chars = max(80, max_chars)
    sentence_parts = re.split(r"(?<=[.!?;:])\s*", text)
    chunks = []
    current = ""

    for part in sentence_parts:
        part = part.strip()
        if not part:
            continue

        if not current:
            current = part
            continue

        if len(current) + 1 + len(part) <= max_chars:
            current = f"{current} {part}"
            continue

        chunks.append(current)
        current = part

    if current:
        chunks.append(current)
    return chunks or [text]


class VoxtralTTS:
    """TTS via Voxtral distant (PC2 style)."""
    
    def __init__(self, url: str = "http://192.168.1.90:9000/tts", 
                 voice: str = "fr_female", speed: float = 1.0,
                 chunk_chars: int = 320, timeout: float = 240.0):
        self.url = url
        self.voice = voice
        self.speed = max(0.5, min(2.0, speed))
        self.chunk_chars = chunk_chars
        self.timeout = timeout
    
    def synthesize(self, text: str) -> bytes:
        """Synthétise du texte en WAV bytes."""
        text = sanitize(text)
        if not text:
            raise ValueError("Texte vide après nettoyage")
        
        chunks = split_text(text, self.chunk_chars)
        wav_parts = []
        
        for chunk in chunks:
            payload = json.dumps({
                "text": chunk,
                "voice": self.voice,
                "speed": self.speed,
            }).encode("utf-8")
            
            req = urllib.request.Request(
                self.url,
                data=payload,
                headers={"Content-Type": "application/json", "Accept": "audio/wav"},
                method="POST",
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                wav_parts.append(resp.read())
        
        return join_wavs(wav_parts)
    
    def synthesize_file(self, text: str, output_path: str, fmt: str = "mp3") -> str:
        """Synthétise et écrit dans un fichier."""
        wav = self.synthesize(text)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        ext = out.suffix.lower().lstrip(".") or "mp3"
        if ext == "wav":
            out.write_bytes(wav)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav)
                tmp_path = tmp.name
            
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("ffmpeg requis pour la conversion")
            
            args = [ffmpeg, "-y", "-loglevel", "error", "-i", tmp_path]
            if ext == "mp3":
                args += ["-ac", "1", "-ar", "24000", "-b:a", "128k"]
            elif ext == "ogg":
                args += ["-acodec", "libopus", "-ac", "1", "-b:a", "64k"]
            args.append(str(out))
            
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            os.unlink(tmp_path)
        
        return str(out)


class LocalTTS:
    """TTS local via Piper (pour le fallback)."""
    
    def __init__(self, model_path: str = None, voice: str = "fr_google_feminine"):
        self.model_path = model_path or os.environ.get("PIPER_MODEL")
        self.voice = voice
    
    def synthesize(self, text: str) -> bytes:
        text = sanitize(text)
        cmd = ["piper", "--sentence_silence", "0.1"]
        if self.model_path:
            cmd += ["--model", self.model_path]
        cmd += ["--output-raw", "-"]
        
        proc = subprocess.run(cmd, input=text, capture_output=True, text=True, check=True)
        return proc.stdout


def join_wavs(wavs: list[bytes], silence_s: float = 0.0) -> bytes:
    """Joint plusieurs chunks WAV."""
    if len(wavs) == 1:
        return wavs[0]
    
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        for i, wav_data in enumerate(wavs):
            with wave.open(io.BytesIO(wav_data), "rb") as src:
                params = src.getparams()
                if i == 0:
                    dst.setparams(params)
                if i > 0 and silence_s > 0:
                    silence = b"\x00" * int(params.framerate * silence_s) * params.nchannels * params.sampwidth
                    dst.writeframes(silence)
                dst.writeframes(src.readframes(params.nframes))
    
    return out.getvalue()
