"""
Memory subsystem — short-term context + long-term persistent storage.

Architecture :
  - ShortTermMemory : contexte actif (RAM), fenêtre de conversation, entités
  - MemoryEvent : event passing entre pipeline et memory manager
  - PriorityFilter : décide quoi stocker via scoring
  - LongTermMemory : ChromaDB (embeddings) + SQLite (metadata)
  - MemoryManager : orchestrateur async
"""
import time
import hashlib
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─── Data models ──────────────────────────────────────────────────────────────

@dataclass
class MemoryEvent:
    """Event passé au MemoryManager via l'event bus."""
    text: str
    source: str  # "user", "assistant", "system", "correction"
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    category: str = "event"  # user_preference, technical_knowledge, event, correction, project_state


@dataclass
class MemoryHit:
    """Résultat de recherche dans la mémoire long-terme."""
    text: str
    score: float  # cosine similarity
    category: str
    timestamp: float


# ─── Short-term memory ────────────────────────────────────────────────────────

class ShortTermMemory:
    """
    Contexte actif de la conversation.
    
    Gère :
    - Historique des turns (max_turns)
    - Entités extraites (noms, projets, outils mentionnés)
    - Context window intelligent pour le LLM
    
    Pas un simple slice fixe — on garde les turns importants
    même au-delà de la limite.
    """
    
    def __init__(self, max_turns: int = 20, max_context_tokens: int = 4000):
        self.max_turns = max_turns
        self.max_context_tokens = max_context_tokens
        self._turns: list[dict] = []  # [{"role": "user"/"assistant", "text": ..., "timestamp": ...}]
        self._entities: dict[str, float] = {}  # entity -> last_seen_timestamp
        self._important_turns: set[int] = set()  # indices de turns importants à garder
    
    def add_turn(self, role: str, text: str, important: bool = False):
        """Ajoute un turn à la mémoire courte."""
        turn = {
            "role": role,
            "text": text,
            "timestamp": time.time(),
        }
        self._turns.append(turn)
        
        if important:
            self._important_turns.add(len(self._turns) - 1)
        
        # Track entities (basic — noms propres, projets)
        self._extract_entities(text)
        
        # Trim old non-important turns
        self._trim()
    
    def _extract_entities(self, text: str):
        """Extraction basique d'entités (noms, projets, outils)."""
        # Simple heuristic : mots en PascalCase ou majuscules
        words = text.split()
        for word in words:
            clean = word.strip(".,;:!?\"'()[]{}")
            if clean.isupper() and len(clean) > 1:
                self._entities[clean] = time.time()
            elif clean.istitle() and len(clean) > 3:
                self._entities[clean] = time.time()
    
    def _trim(self):
        """Supprime les anciens turns non-importants pour respecter max_turns."""
        if len(self._turns) <= self.max_turns:
            return
        
        # Garde les turns importants, supprime les autres du début
        while len(self._turns) > self.max_turns:
            idx_to_remove = 0
            # Saute les turns importants
            while idx_to_remove in self._important_turns and idx_to_remove < len(self._turns):
                idx_to_remove += 1
            
            if idx_to_remove in self._important_turns:
                break  # Pas de quoi supprimer
            
            self._turns.pop(idx_to_remove)
            # Ajuste les indices importants
            self._important_turns = {
                i - 1 if i > idx_to_remove else i 
                for i in self._important_turns
            }
    
    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        Récupère le contexte formaté pour le LLM.
        
        Strategy : on prend les turns récents + les turns importants,
        on respecte max_context_tokens.
        """
        limit = max_tokens or self.max_context_tokens
        result_parts = []
        total_tokens = 0
        
        # Trie par timestamp, les plus récents en premier
        sorted_turns = sorted(
            enumerate(self._turns),
            key=lambda x: x[1]["timestamp"],
            reverse=True
        )
        
        for idx, turn in sorted_turns:
            text = turn["text"]
            role = turn["role"]
            tokens = len(text) // 4  # rough estimate
            label = "Ali" if role == "user" else "Zaza"
            
            if total_tokens + tokens > limit:
                break
            
            result_parts.insert(0, f"{label}: {text}")
            total_tokens += tokens
        
        return "\n\n".join(result_parts)
    
    def get_turns(self) -> list[dict]:
        """Retourne tous les turns (pour debug/inspection)."""
        return list(self._turns)
    
    def clear(self):
        """Vide la mémoire courte (nouvelle session)."""
        self._turns.clear()
        self._entities.clear()
        self._important_turns.clear()
    
    @property
    def turn_count(self) -> int:
        return len(self._turns)
    
    @property
    def entities(self) -> dict[str, float]:
        return dict(self._entities)


# ─── Priority filter ──────────────────────────────────────────────────────────

class PriorityFilter:
    """
    Décide si un event mérite d'être stocké en mémoire long-terme.
    
    Score : 0.0 (jamais) → 1.0 (toujours)
    Threshold par défaut : 0.3
    """
    
    # Patterns hautes priorités
    HIGH_PATTERNS = [
        r"préfère|préférence|toujours|jamais|n'aime pas|adore",  # Préférences
        r"rappelle-toi|retiens|mémorise|note bien",  # Instruction explicite de mémoriser
        r"corrig|erreur|faux|pas vrai|c'est pas",  # Corrections
        r"décidé|choisi|option|solution|final",  # Décisions
    ]
    
    MEDIUM_PATTERNS = [
        r"install|config|setup|build|déploy",  # Infos techniques
        r"version|release|mise à jour",  # State du projet
        r"erreur|bug|fix|patch|workaround",  # Problèmes rencontrés
    ]
    
    LOW_PATTERNS = [
        r"bonjour|salut|ça va|merci|ok|oui|non|d'accord|super",  # Banalités
        r"comment tu t'appelles|qui es-tu|c'est quoi",  # Questions basiques
    ]
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self._compile_patterns()
    
    def _compile_patterns(self):
        import re
        self._high = [re.compile(p, re.I) for p in self.HIGH_PATTERNS]
        self._medium = [re.compile(p, re.I) for p in self.MEDIUM_PATTERNS]
        self._low = [re.compile(p, re.I) for p in self.LOW_PATTERNS]
    
    def score(self, event: MemoryEvent) -> float:
        """
        Score un event.
        
        Returns float 0.0-1.0.
        """
        text = event.text.lower()
        
        # Source weighting
        source_bonus = {
            "correction": 0.4,
            "user": 0.1,
            "assistant": 0.0,
            "system": 0.2,
        }.get(event.source, 0.0)
        
        # Pattern matching
        pattern_score = 0.0
        for pat in self._high:
            if pat.search(text):
                pattern_score = max(pattern_score, 0.7)
        for pat in self._medium:
            if pat.search(text):
                pattern_score = max(pattern_score, 0.4)
        for pat in self._low:
            if pat.search(text):
                pattern_score = min(pattern_score, 0.1)
        
        # Category bonus
        category_bonus = {
            "user_preference": 0.3,
            "correction": 0.3,
            "technical_knowledge": 0.15,
            "project_state": 0.15,
            "event": 0.0,
        }.get(event.category, 0.0)
        
        final = min(1.0, pattern_score + source_bonus + category_bonus)
        return final
    
    def should_store(self, event: MemoryEvent) -> bool:
        """Doit-on stocker cet event en mémoire long-terme ?"""
        return self.score(event) >= self.threshold
    
    def is_duplicate(self, text: str, existing_embeddings: Optional[list[np.ndarray]] = None, 
                     existing_threshold: float = 0.95) -> bool:
        """
        Check si le text est trop similaire à un item existant (via embeddings).
        
        Pour l'instant : hash simple. Sera remplacé par cosine similarity
        quand les embeddings seront disponibles.
        """
        if not existing_embeddings:
            return False
        
        # Placeholder — sera implémenté avec le real embedding check
        # Pour l'instant on retourne False (pas de dedup)
        return False
