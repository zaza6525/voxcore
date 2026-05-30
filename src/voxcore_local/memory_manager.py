"""
Memory Manager — orchestrateur de la mémoire.

Connecte le pipeline vocal au système de mémoire :
- Reçoit les events du pipeline
- Applique le priority filter
- Stocke en long-term si pertinent
- Injecte le contexte dans les prompts LLM
"""
import time
import threading
from typing import Optional
from .memory import MemoryEvent, MemoryHit, ShortTermMemory, PriorityFilter
from .memory_store import LongTermMemory


class MemoryManager:
    """
    Orchestrateur principal de la mémoire.
    
    Usage dans le pipeline :
    
        memory = MemoryManager()
        
        # Quand l'utilisateur parle :
        event = MemoryEvent(text="...", source="user")
        memory.process(event)
        
        # Avant d'envoyer au LLM :
        context = memory.build_context(user_message="...")
        prompt = f"{context}\\n\\nAli: {user_message}\\n\\nZaza: "
    
    Thread-safe pour usage async dans le pipeline vocal.
    """
    
    def __init__(self, 
                 db_path: Optional[str] = None,
                 chroma_path: Optional[str] = None,
                 short_term_turns: int = 20,
                 filter_threshold: float = 0.3):
        self.short_term = ShortTermMemory(max_turns=short_term_turns)
        self.filter = PriorityFilter(threshold=filter_threshold)
        self.long_term = LongTermMemory(
            db_path=db_path or "~/.local/share/voxcore/memory.db",
            chroma_path=chroma_path or "~/.local/share/voxcore/chroma_memory"
        )
        self._lock = threading.Lock()
        self._session_start = time.time()
    
    def process(self, event: MemoryEvent):
        """
        Traite un event entrant.
        
        1. Ajoute à la mémoire court-terme
        2. Si score élevé → stocke en long-terme
        """
        with self._lock:
            # Court-terme : toujours
            important = event.source == "correction" or event.category == "user_preference"
            self.short_term.add_turn(event.source, event.text, important=important)
            
            # Long-terme : si le filter dit oui
            if self.filter.should_store(event):
                event.metadata["score"] = self.filter.score(event)
                self.long_term.store(event)
    
    def build_context(self, query: str, 
                      short_term_only: bool = False,
                      long_term_limit: int = 3) -> str:
        """
        Construit le contexte pour le LLM.
        
        Combine :
        1. Mémoire court-terme (conversation active)
        2. Mémoire long-terme (rappel sémantique si query le mérite)
        """
        parts = []
        
        # Court-terme : conversation active
        st_context = self.short_term.get_context()
        if st_context:
            parts.append(f"### Conversation en cours\n{st_context}")
        
        # Long-terme : si la query mérite un rappel
        if not short_term_only:
            lt_hits = self.long_term.query(query, limit=long_term_limit)
            if lt_hits:
                recalled = []
                for hit in lt_hits:
                    recalled.append(f"- {hit.text}")
                parts.append(f"### Mémoire (rappel pertinent)\n" + "\n".join(recalled))
        
        return "\n\n".join(parts)
    
    def recall(self, query: str, limit: int = 3, 
               category: Optional[str] = None) -> list[MemoryHit]:
        """
        Recherche explicite dans la mémoire long-terme.
        
        Utilisé quand l'assistant a besoin de vérifier un fait.
        """
        with self._lock:
            return self.long_term.query(query, limit=limit, category=category)
    
    def record_preference(self, text: str):
        """Shortcut : enregistre explicitement une préférence utilisateur."""
        event = MemoryEvent(
            text=text,
            source="user",
            category="user_preference",
        )
        self.process(event)
    
    def record_correction(self, text: str):
        """Shortcut : enregistre explicitement une correction."""
        event = MemoryEvent(
            text=text,
            source="correction",
            category="correction",
        )
        self.process(event)
    
    def get_stats(self) -> dict:
        """Statistiques pour debug/monitoring."""
        return {
            "short_term_turns": self.short_term.turn_count,
            "short_term_entities": len(self.short_term.entities),
            "long_term_count": self.long_term.count(),
            "session_duration": time.time() - self._session_start,
        }
    
    def new_session(self):
        """Nouvelle session — vide court-terme, garde long-terme."""
        with self._lock:
            self.short_term.clear()
            self._session_start = time.time()
