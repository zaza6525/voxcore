"""TTS Router — sélection et fallback entre backends TTS.

Gère plusieurs backends TTS avec fallback automatique.
Si Voxtral PC2 est down, bascule sur Piper local.
"""
import os
import tempfile
import subprocess
import urllib.request
import json
import shutil
from typing import Optional
from pathlib import Path

from .tts import VoxtralTTS, sanitize, join_wavs


class TTSRouter:
    """Route les requêtes TTS vers le backend disponible.
    
    Ordre de priorité :
    1. Voxtral PC2 (configuré via config.yaml)
    2. Piper local (si installé)
    3. Erreur
    """
    
    def __init__(self, config: dict):
        tts_cfg = config["tts"]
        
        # Backend primaire : Voxtral
        self.primary = VoxtralTTS(
            url=tts_cfg["url"],
            voice=tts_cfg["voice"],
            speed=tts_cfg["speed"],
            chunk_chars=tts_cfg["chunk_chars"],
        )
        self.primary_name = "voxtral"
        
        # Backend secondaire : Piper (si disponible)
        self._piper_model = tts_cfg.get("piper_model") or os.environ.get("PIPER_MODEL")
        self._piper_voice = tts_cfg.get("piper_voice", "fr_google_feminine")
        self._piper_available = shutil.which("piper") is not None
        
        # State
        self._active = self.primary_name
        self._error_count = 0
        self.max_errors = 3  # Après 3 erreurs, on bascule
    
    def health_check(self) -> dict:
        """Vérifie la santé des backends."""
        result = {}
        
        # Voxtral
        health_url = self.primary.url.replace("/tts", "/health")
        try:
            with urllib.request.urlopen(health_url, timeout=5) as r:
                result["voxtral"] = {"status": "ok", "detail": r.read().decode()}
        except Exception as e:
            result["voxtral"] = {"status": "error", "detail": str(e)}
        
        # Piper
        if self._piper_available:
            try:
                proc = subprocess.run(
                    ["piper", "--help"], capture_output=True, timeout=3
                )
                result["piper"] = {"status": "ok", "model": self._piper_model}
            except Exception as e:
                result["piper"] = {"status": "error", "detail": str(e)}
        else:
            result["piper"] = {"status": "not_installed"}
        
        return result
    
    def synthesize(self, text: str) -> bytes:
        """Synthétise avec fallback automatique."""
        text = sanitize(text)
        if not text:
            raise ValueError("Texte vide")
        
        # Essaie le backend actif
        if self._active == "voxtral":
            try:
                return self.primary.synthesize(text)
            except Exception as e:
                self._error_count += 1
                if self._error_count >= self.max_errors and self._piper_available:
                    print(f"⚠️  Voxtral down ({e}), fallback Piper")
                    self._active = "piper"
                    return self._synthesize_piper(text)
                raise
        
        elif self._active == "piper":
            try:
                return self._synthesize_piper(text)
            except Exception as e:
                # Réessaie Voxtral
                self._error_count = max(0, self._error_count - 1)
                if self._error_count == 0:
                    self._active = "voxtral"
                raise e
        
        raise RuntimeError("Aucun backend TTS disponible")
    
    def synthesize_file(self, text: str, output_path: str, fmt: str = "mp3") -> str:
        """Synthétise et écrit dans un fichier."""
        wav = self.synthesize(text)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        
        ext = out.suffix.lower().lstrip(".") or fmt
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
    
    def _synthesize_piper(self, text: str) -> bytes:
        """Synthèse via Piper local."""
        cmd = ["piper", "--sentence_silence", "0.1"]
        if self._piper_model:
            cmd += ["--model", self._piper_model]
        cmd += ["--output-raw", "-"]
        
        proc = subprocess.run(
            cmd, input=text, capture_output=True, text=True, check=True
        )
        
        # Convertit raw PCM 22050Hz en WAV
        import io
        import wave
        import struct
        
        pcm_bytes = proc.stdout.encode('latin1') if isinstance(proc.stdout, str) else proc.stdout
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(pcm_bytes)
        
        return wav_buf.getvalue()
    
    @property
    def active_backend(self) -> str:
        return self._active
    
    def reset(self):
        """Reset l'état des erreurs."""
        self._error_count = 0
        self._active = self.primary_name
