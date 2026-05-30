"""Pipeline STT → LLM → TTS complet avec mémoire."""
from pathlib import Path
from typing import Optional
from .stt import STTEngine
from .llm import LLMEngine
from .tts import VoxtralTTS
from .config import load_config
from .memory_manager import MemoryManager
from .memory import MemoryEvent

class VoicePipeline:
    """Pipeline complet : audio → réponse vocale."""
    
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
        
        self.system_prompt = cfg["llm"]["system_prompt"]
        self.max_tokens = cfg["llm"]["max_tokens"]
        self.output_fmt = cfg["output"]["format"]
        self.output_dir = Path(cfg["output"]["dir"])
        
        # Memory system
        memory_cfg = cfg.get("memory", {})
        self.memory = MemoryManager(
            db_path=memory_cfg.get("db_path"),
            chroma_path=memory_cfg.get("chroma_path"),
            short_term_turns=memory_cfg.get("short_term_turns", 20),
            filter_threshold=memory_cfg.get("filter_threshold", 0.3),
        )
    
    def run(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """Pipeline complet : fichier audio → fichier vocal de réponse.
        
        Flux : STT → Memory (store + context) → LLM → Memory (store) → TTS
        
        Returns:
            Le texte de la réponse LLM.
        """
        # 1. STT
        text = self.stt.transcribe(audio_path)
        if not text:
            raise ValueError("STT : pas de texte détecté")
        
        # 2. Memory : stocke l'input utilisateur + construit le contexte
        self.memory.process(MemoryEvent(text=text, source="user"))
        memory_context = self.memory.build_context(query=text)
        
        # 3. LLM avec contexte mémoire
        full_system = self.system_prompt
        if memory_context:
            full_system = f"{self.system_prompt}\n\n{memory_context}"
        
        response = self.llm.complete(
            text, 
            system=full_system, 
            max_tokens=self.max_tokens
        )
        
        # 4. Memory : stocke la réponse
        self.memory.process(MemoryEvent(text=response, source="assistant"))
        
        # 5. TTS
        if not output_path:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(self.output_dir / f"response.{self.output_fmt}")
        
        self.tts.synthesize_file(response, output_path, fmt=self.output_fmt)
        
        return response
    
    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> str:
        """Juste TTS : texte → fichier audio."""
        if not output_path:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(self.output_dir / f"tts.{self.output_fmt}")
        
        self.tts.synthesize_file(text, output_path, fmt=self.output_fmt)
        return output_path
    
    def transcribe(self, audio_path: str) -> str:
        """Juste STT : audio → texte."""
        return self.stt.transcribe(audio_path)
    
    def chat(self, text: str) -> str:
        """Chat text-to-text avec mémoire.
        
        Stocke l'input, injecte le contexte mémoire, appelle le LLM, stocke la réponse.
        """
        self.memory.process(MemoryEvent(text=text, source="user"))
        memory_context = self.memory.build_context(query=text)
        
        full_system = self.system_prompt
        if memory_context:
            full_system = f"{self.system_prompt}\n\n{memory_context}"
        
        response = self.llm.complete(text, system=full_system, max_tokens=self.max_tokens)
        self.memory.process(MemoryEvent(text=response, source="assistant"))
        return response
    
    def memory_stats(self) -> dict:
        """Statistiques de la mémoire."""
        return self.memory.get_stats()
