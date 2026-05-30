"""Mode live : microphone → STT → LLM → TTS → speaker.

Boucle principale de VoxCore en temps réel.
"""
import logging
import tempfile
import threading
from typing import Optional

from .stt import STTEngine
from .llm import LLMEngine
from .tts import VoxtralTTS, sanitize
from .microphone import MicrophoneCapture
from .audio_player import AudioPlayer
from .config import load_config

logger = logging.getLogger(__name__)


class LiveSession:
    """Session live : écoute → répond vocalement."""
    
    def __init__(self, config: Optional[dict] = None):
        cfg = config or load_config()
        
        self.stt = STTEngine(
            model_size=cfg["stt"]["model_size"],
            device=cfg["stt"]["device"],
            language=cfg["stt"]["language"],
        )
        
        self.llm = LLMEngine(
            base_url=cfg["llm"]["base_url"],
            model=cfg["llm"]["model"],
            api_key=cfg["llm"]["api_key"],
        )
        
        self.tts = VoxtralTTS(
            url=cfg["tts"]["url"],
            voice=cfg["tts"]["voice"],
            speed=cfg["tts"]["speed"],
            chunk_chars=cfg["tts"]["chunk_chars"],
        )
        
        self.player = AudioPlayer()
        self.system_prompt = cfg["llm"]["system_prompt"]
        self.max_tokens = cfg["llm"]["max_tokens"]
        
        # State
        self._running = False
        self._listening = False
        self._responding = False
        self._history = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def start(self, mic_device: Optional[int] = None, 
              vad_threshold: float = 0.01,
              min_silence: float = 0.8):
        """Démarre la session live."""
        self._running = True
        
        mic = MicrophoneCapture(
            device_index=mic_device,
            vad_threshold=vad_threshold,
            min_silence=min_silence,
        )
        
        print(f"🎙  VoxCore LIVE — écoute (Ctrl+C pour arrêter)")
        print(f"   Langue: {self.stt.language}")
        print(f"   LLM: {self.llm.base_url}")
        print(f"   TTS: {self.tts.url}")
        print()
        
        def on_phrase(wav_bytes: bytes):
            if not self._running or self._responding:
                return
            
            try:
                self._handle_phrase(wav_bytes)
            except Exception as e:
                logger.error(f"Erreur pipeline: {e}", exc_info=True)
        
        mic.start(on_phrase)
        
        try:
            while self._running:
                import time
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n⏹  Arrêt...")
        finally:
            self.stop()
            mic.stop()
    
    def _handle_phrase(self, wav_bytes: bytes):
        """Traite une phrase détectée."""
        self._listening = True
        print("🔴 [transcription...]", end='\r', flush=True)
        
        # 1. STT
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name
        
        try:
            text = self.stt.transcribe(wav_path)
        finally:
            import os
            os.unlink(wav_path)
        
        if not text:
            self._listening = False
            print(" " * 40 + "\r", end='', flush=True)
            return
        
        print(f"\r🗣  Vous : {text}")
        self._listening = False
        
        # Ajoute au historique
        self._history.append({"role": "user", "content": text})
        
        # 2. LLM
        self._responding = True
        print("🤖 [réflexion...]", flush=True)
        
        try:
            response = self.llm.chat(self._history, max_tokens=self.max_tokens)
            self._history.append({"role": "assistant", "content": response})
            
            # Garne le historique à 20 messages max
            while len(self._history) > 21:  # 1 system + 20 exchange
                self._history.pop(1)
        except Exception as e:
            print(f"❌ Erreur LLM: {e}")
            self._responding = False
            return
        
        print(f"💬 Zaza : {response}")
        
        # 3. TTS + lecture
        print("🔊 [synthèse vocale...]", flush=True)
        try:
            audio = self.tts.synthesize(sanitize(response))
            self.player.play_bytes(audio, format='WAV')
            print(f"✅ [lu] " + " " * 30, end='\r', flush=True)
        except Exception as e:
            print(f"❌ Erreur TTS: {e}")
        finally:
            self._responding = False
    
    def stop(self):
        """Arrête la session."""
        self._running = False
