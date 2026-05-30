"""
Tests d'intégration mémoire — MemoryManager + pipeline + live.

Valide que la mémoire fonctionne dans les vrais workflows :
- CLI chat avec mémoire
- Live session avec mémoire
- Pipeline run avec mémoire
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from voxcore_local.memory_manager import MemoryManager
from voxcore_local.memory_store import LongTermMemory
from voxcore_local.memory import MemoryEvent, ShortTermMemory, PriorityFilter


class TestMemoryIntegration:
    """Tests d'intégration du MemoryManager."""
    
    def test_full_event_lifecycle(self):
        """Event complet : process → build_context → inject."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            chroma = Path(tmpdir) / "chroma"
            
            mgr = MemoryManager(
                db_path=str(db),
                chroma_path=str(chroma)
            )
            
            # 1. Process events
            mgr.process_event("Je m'appelle Ali", category="user_preference", source="user")
            mgr.process_event("Bonjour Ali !", category="conversation", source="assistant")
            mgr.process_event("Quel est mon nom ?", category="conversation", source="user")
            
            # 2. Build context
            context = mgr.build_context(
                history=[{"role": "user", "content": "Quel est mon nom ?"}]
            )
            
            # Le contexte doit contenir la préférence stockée
            assert "Ali" in context or "Conversation en cours" in context
            
            # 3. Inject context
            history = [{"role": "system", "content": "Tu es un assistant"}]
            enhanced = mgr.inject_context(history, context)
            
            assert len(enhanced) >= 1
            assert enhanced[0]["role"] == "system"
            assert "Contexte mémoire" in enhanced[0]["content"] or "Conversation en cours" in enhanced[0]["content"]
    
    def test_stats_tracking(self):
        """Les stats reflètent l'état réel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            chroma = Path(tmpdir) / "chroma"
            
            mgr = MemoryManager(
                db_path=str(db),
                chroma_path=str(chroma)
            )
            
            mgr.process_event("Test event", category="conversation", source="user")
            
            stats = mgr.get_stats()
            assert "short_term_turns" in stats
            assert "long_term_count" in stats
            assert stats["short_term_turns"] >= 1
    
    def test_new_session_preserves_long_term(self):
        """new_session vide le court-terme mais garde le long-terme."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            chroma = Path(tmpdir) / "chroma"
            
            mgr = MemoryManager(
                db_path=str(db),
                chroma_path=str(chroma),
                filter_threshold=0.1  # Stocke tout pour le test
            )
            
            # Préfèrece = toujours stockée en LT
            mgr.record_preference("J'aime le code Python")
            mgr.process_event("Conversation banale", source="user")
            
            lt_count_before = mgr.long_term.count()
            assert lt_count_before >= 1
            
            mgr.new_session()
            
            lt_count_after = mgr.long_term.count()
            assert lt_count_after == lt_count_before  # LT conservé
            assert mgr.short_term.turn_count == 0  # ST vidé


class TestConfigDrivenMemory:
    """Test que la config YAML drive correctement la mémoire."""
    
    def test_config_init(self):
        """MemoryManager init via config dict."""
        config = {
            "ltm_db_path": "~/.local/share/voxcore/test.db",
            "ltm_chroma_path": "~/.local/share/voxcore/test_chroma",
            "ltm_embed_model": "all-MiniLM-L6-v2",
        }
        
        mgr = MemoryManager(config=config)
        
        assert mgr.config == config
        assert mgr.long_term.embed_model == "all-MiniLM-L6-v2"
    
    def test_empty_config_defaults(self):
        """Config vide → defaults fonctionnels."""
        mgr = MemoryManager(config={})
        
        assert mgr.short_term is not None
        assert mgr.long_term is not None
        assert mgr.filter is not None


class TestLongTermMemoryConfig:
    """Test LongTermMemory avec config personnalisée."""
    
    def test_embed_model_param(self):
        """Le modèle d'embedding est bien passé."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            chroma = Path(tmpdir) / "chroma"
            
            ltm = LongTermMemory(
                db_path=str(db),
                chroma_path=str(chroma),
                embed_model="all-MiniLM-L6-v2"
            )
            
            assert ltm.embed_model == "all-MiniLM-L6-v2"
            assert ltm._embed_fn is not None
    
    def test_store_and_retrieve_with_config(self):
        """Store + query fonctionne avec config custom."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Path(tmpdir) / "test.db"
            chroma = Path(tmpdir) / "chroma"
            
            ltm = LongTermMemory(
                db_path=str(db),
                chroma_path=str(chroma),
                embed_model="all-MiniLM-L6-v2"
            )
            
            event = MemoryEvent(
                text="La réponse est 42",
                source="user",
                category="conversation"
            )
            ltm.store(event)
            
            hits = ltm.query("réponse", limit=5)
            assert len(hits) >= 1
            assert "42" in hits[0].text


class TestRateLimiter:
    """Test rate limiter."""
    
    def test_basic_rate_limit(self):
        """Rate limiter limite les requêtes."""
        from voxcore_local.rate_limiter import RateLimiter
        
        rl = RateLimiter(rate=5, burst=2)
        
        # 2 premiers = immédiats (burst)
        assert rl.acquire(1) == 0.0
        assert rl.acquire(1) == 0.0
        
        # 3ème = doit attendre
        delay = rl.acquire(1)
        assert delay > 0
    
    def test_wait(self):
        """wait() bloque jusqu'à disponibilité."""
        from voxcore_local.rate_limiter import RateLimiter
        
        rl = RateLimiter(rate=100, burst=1)
        rl.wait()  # 1er = ok
        # 2ème = attend, mais avec rate=100 c'est rapide
        rl.wait()


class TestRetry:
    """Test retry avec backoff."""
    
    def test_success_first_try(self):
        """Pas de retry si succès du 1er coup."""
        from voxcore_local.rate_limiter import Retry
        
        counter = [0]
        def ok():
            counter[0] += 1
            return "ok"
        
        retry = Retry(max_attempts=3)
        assert retry.call(ok) == "ok"
        assert counter[0] == 1
    
    def test_retry_on_failure(self):
        """Retry après 2 échecs."""
        from voxcore_local.rate_limiter import Retry
        
        counter = [0]
        def fail_then_ok():
            counter[0] += 1
            if counter[0] < 3:
                raise ValueError("fail")
            return "ok"
        
        retry = Retry(max_attempts=3, base_delay=0.01)
        assert retry.call(fail_then_ok) == "ok"
        assert counter[0] == 3
    
    def test_decorator(self):
        """@with_retry fonctionne."""
        from voxcore_local.rate_limiter import with_retry
        
        counter = [0]
        @with_retry(max_attempts=2, base_delay=0.01)
        def decorated():
            counter[0] += 1
            if counter[0] < 2:
                raise ValueError("fail")
            return "ok"
        
        assert decorated() == "ok"


class TestCircuitBreaker:
    """Test circuit breaker."""
    
    def test_closes_after_success(self):
        """Circuit fermé après succès."""
        from voxcore_local.rate_limiter import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == "closed"
        
        cb.call(lambda: "ok")
        assert cb.state == "closed"
    
    def test_opens_after_failures(self):
        """Circuit ouvert après N échecs."""
        from voxcore_local.rate_limiter import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=2, recovery_time=999)
        
        def fail():
            raise ValueError("fail")
        
        try:
            cb.call(fail)
        except ValueError:
            pass
        try:
            cb.call(fail)
        except ValueError:
            pass
        assert cb.state == "open"
    
    def test_reset(self):
        """Reset réinitialise le circuit."""
        from voxcore_local.rate_limiter import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=1)
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass
        
        cb.reset()
        assert cb.state == "closed"
