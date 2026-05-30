"""
Long-term memory storage — SQLite + ChromaDB.

SQLite : métadonnées structurées (timestamp, category, score, source, raw text)
ChromaDB : embeddings pour la recherche sémantique

Les deux sont synchronisés — un insert dans l'un = insert dans l'autre.
"""
import sqlite3
import uuid
import time
from pathlib import Path
from typing import Optional
from .memory import MemoryEvent, MemoryHit


class LongTermMemory:
    """
    Stockage persistant de la mémoire long-terme.
    
    SQLite pour les filtres structqués, ChromaDB pour la recherche sémantique.
    Les deux pointent sur le même documentID.
    """
    
    def __init__(self, db_path: str = "~/.local/share/voxcore/memory.db",
                 chroma_path: str = "~/.local/share/voxcore/chroma_memory",
                 embed_model: str = "all-MiniLM-L6-v2",
                 embedding_dim: int = 384):
        self.db_path = str(Path(db_path).expanduser())
        self.chroma_path = str(Path(chroma_path).expanduser())
        self.embed_model = embed_model
        self.embedding_dim = embedding_dim
        
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.chroma_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._init_sqlite()
        self._chroma_client = None
        self._chroma_collection = None
        self._embed_fn = None
        self._init_chroma()
    
    def _init_sqlite(self):
        """Créate/upgrade la base SQLite."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'user',
                category TEXT NOT NULL DEFAULT 'event',
                score REAL NOT NULL DEFAULT 0.5,
                timestamp REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_score ON memories(score)
        """)
        conn.commit()
        conn.close()
    
    def _init_chroma(self):
        """Initialise ChromaDB (lazy, seulement quand on a un embedder)."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            self._chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            # On utilise sentence-transformers all-MiniLM-L6-v2 (384D) par défaut
            # — léger, rapide, pas besoin de GPU
            self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embed_model
            )
            
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name="voxcore_memory",
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            # ChromaDB pas disponible — mode SQLite-only
            self._chroma_client = None
            self._chroma_collection = None
            self._embed_fn = None
    
    def store(self, event: MemoryEvent) -> str:
        """
        Stocke un event en mémoire long-terme.
        
        Retourne le document ID.
        """
        doc_id = str(uuid.uuid4())
        
        # SQLite
        import json
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO memories VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, event.text, event.source, event.category, 
             event.metadata.get("score", 0.5), event.timestamp,
             json.dumps(event.metadata))
        )
        conn.commit()
        conn.close()
        
        # ChromaDB
        if self._chroma_collection:
            try:
                self._chroma_collection.add(
                    documents=[event.text],
                    ids=[doc_id],
                    metadatas=[{
                        "source": event.source,
                        "category": event.category,
                        "timestamp": event.timestamp,
                        "score": event.metadata.get("score", 0.5),
                    }]
                )
            except Exception:
                # Si Chroma échoue, on garde SQLite — pas critique
                pass
        
        return doc_id
    
    def query(self, query_text: str, limit: int = 3, 
              category: Optional[str] = None,
              min_score: float = 0.3,
              since: Optional[float] = None) -> list[MemoryHit]:
        """
        Recherche sémantique + filtres.
        
        Strategy :
        1. Si ChromaDB dispo : recherche sémantique, puis filtre par catégorie/date
        2. Sinon : recherche texte plein via SQLite (LIKE)
        """
        if self._chroma_collection:
            return self._query_chroma(query_text, limit, category, min_score, since)
        else:
            return self._query_sqlite(query_text, limit, category, min_score, since)
    
    def _query_chroma(self, query_text: str, limit: int, 
                       filter_category: Optional[str] = None,
                       min_score: float = 0.0,
                       since: Optional[float] = None) -> list[MemoryHit]:
        """Recherche sémantique via ChromaDB."""
        results = self._chroma_collection.query(
            query_texts=[query_text],
            n_results=limit * 3,  # Récupère plus pour filtrer après
        )
        
        hits = []
        if results["documents"]:
            for doc, dist, meta, doc_id in zip(
                 results["documents"][0],
                 results["distances"][0],
                 results["metadatas"][0],
                 results["ids"][0]
             ):
                 # Cosine distance → similarity
                 similarity = 1.0 - float(dist)
                    
                 # Apply filters
                 cat = str(meta.get("category", "event"))
                 ts = float(meta.get("timestamp", 0))
                 if filter_category and cat != filter_category:
                      continue
                 if since and ts < since:
                     continue
                 if similarity < min_score:
                     continue
                    
                 hits.append(MemoryHit(
                     text=str(doc),
                     score=similarity,
                     category=cat,
                     timestamp=ts,
                 ))
        
        # Sort by similarity, limit
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]
    
    def _query_sqlite(self, query_text: str, limit: int, category, min_score, since) -> list[MemoryHit]:
        """Fallback : recherche LIKE via SQLite."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT id, text, source, category, score, timestamp FROM memories WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit * 3)
        
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        # Score texte simple : contient-il les mots-clés ?
        words = query_text.lower().split()
        hits = []
        for row_id, text, source, cat, score, ts in rows:
            # Simple keyword overlap
            text_lower = text.lower()
            match_count = sum(1 for w in words if w in text_lower and len(w) > 2)
            text_score = match_count / max(len(words), 1)
            
            if text_score >= min_score:
                hits.append(MemoryHit(
                    text=text,
                    score=text_score,
                    category=cat,
                    timestamp=ts,
                ))
        
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]
    
    def count(self) -> int:
        """Nombre total de memories stockées."""
        conn = sqlite3.connect(self.db_path)
        result = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        return result
    
    def delete_by_id(self, doc_id: str):
        """Supprime une mémoire par ID."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM memories WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()
        
        if self._chroma_collection:
            try:
                self._chroma_collection.delete(ids=[doc_id])
            except Exception:
                pass
    
    def clear(self):
        """Vide tout (reset complet)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM memories")
        conn.commit()
        conn.close()
        
        if self._chroma_collection:
            try:
                # Supprime tous les IDs
                all_ids = self._chroma_collection.get()["ids"]
                if all_ids:
                    self._chroma_collection.delete(ids=all_ids)
            except Exception:
                pass
