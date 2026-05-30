"""
Memory Manager — orchestrateur de la mémoire.

Connecte le pipeline vocal au système de mémoire :
- Reçoit les events du pipeline
- Applique le priority filter
- Stocke en long-term si pertinent
- Injecte le contexte dans les prompts LLM
"""
from __future__ import annotations
import time
import threading
from typing import Optional, List, Any
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
    
    short_term: ShortTermMemory
    filter: PriorityFilter
    long_term: LongTermMemory
    config: dict[str, Any]
    
    def __init__(self, 
                 config: Optional[dict] = None,
                 db_path: Optional[str] = None,
                 chroma_path: Optional[str] = None,
                 short_term_turns: int = 20,
                 filter_threshold: float = 0.3):
        self.config = config or {}
        self.short_term = ShortTermMemory(max_turns=short_term_turns)
        self.filter = PriorityFilter(threshold=filter_threshold)
        self.long_term = LongTermMemory(
            db_path=db_path or self.config.get("ltm_db_path", "~/.local/share/voxcore/memory.db"),
            chroma_path=chroma_path or self.config.get("ltm_chroma_path", "~/.local/share/voxcore/chroma_memory"),
            embed_model=self.config.get("ltm_embed_model", "all-MiniLM-L6-v2")
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
    
    def process_event(self, content: str, category: str = "conversation", source: str = "user"):
        """Shortcut : crée et traite un MemoryEvent en une ligne."""
        event = MemoryEvent(text=content, source=source, category=category)
        self.process(event)
    
    def build_context(self,
                      history: Optional[List[dict] | str] = None,
                      query: str = "",
                      short_term_only: bool = False,
                      top_k: int = 3) -> str:
        """
        Construit le contexte pour le LLM.
        
        Args:
            history: Soit une liste de dicts messages, soit un string query (backward compat)
            query: Query explicite pour la recherche sémantique
            short_term_only: Ne cherche pas en long-terme
            top_k: Nombre de hits long-terme à inclure
        """
        # Backward compat : si history est un string, c'est une query
        if isinstance(history, str):
            query = query or history
            history = None
        
        # Si pas de query explicite, prend le dernier message user
        if not query and history:
            for msg in reversed(history):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    query = msg.get("content", "")
                    break
        
        parts = []
        
        # Court-terme : conversation active
        st_context = self.short_term.get_context()
        if st_context:
            parts.append(f"### Conversation en cours\n{st_context}")
        
        # Long-terme : si la query mérite un rappel
        if query and not short_term_only:
            lt_hits = self.long_term.query(query, limit=top_k)
            if lt_hits:
                recalled = []
                for hit in lt_hits:
                    recalled.append(f"- {hit.text}")
                parts.append(f"### Mémoire (rappel pertinent)\n" + "\n".join(recalled))
        
        return "\n\n".join(parts)
    
    def inject_context(self, history: List[dict], context: str) -> List[dict]:
        """
        Injecte le contexte mémoire dans le début de l'historique messages.
        
        Ajoute un message system après le premier avec le contexte.
        """
        if not context:
            return history
        
        new_history = list(history)
        # Insère après le premier message (généralement le system prompt)
        system_context = f"\n\nContexte mémoire :\n{context}"
        if new_history and new_history[0].get("role") == "system":
            new_history[0] = {**new_history[0], "content": new_history[0]["content"] + system_context}
        else:
            new_history.insert(0, {"role": "system", "content": f"Contexte mémoire :\n{context}"})
        
        return new_history
    
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
