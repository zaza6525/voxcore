"""Tests mémoire : short-term, priority filter, long-term storage."""
import pytest
import tempfile
import os
from voxcore_local.memory import ShortTermMemory, PriorityFilter, MemoryEvent
from voxcore_local.memory_store import LongTermMemory
from voxcore_local.memory_manager import MemoryManager


# ─── Short-term memory ───────────────────────────────────────────────────────

class TestShortTermMemory:
    
    def test_add_and_get_turns(self):
        mem = ShortTermMemory(max_turns=5)
        mem.add_turn("user", "Bonjour")
        mem.add_turn("assistant", "Salut !")
        assert mem.turn_count == 2
        turns = mem.get_turns()
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"
    
    def test_trim_old_turns(self):
        mem = ShortTermMemory(max_turns=3)
        for i in range(5):
            mem.add_turn("user", f"Message {i}")
        # Doit garder max 3
        assert mem.turn_count <= 3
    
    def test_important_turns_kept(self):
        mem = ShortTermMemory(max_turns=3)
        mem.add_turn("user", "Important", important=True)
        for i in range(4):
            mem.add_turn("user", f"Banal {i}")
        # Le turn important doit être présent
        turns = mem.get_turns()
        texts = [t["text"] for t in turns]
        assert "Important" in texts
    
    def test_context_generation(self):
        mem = ShortTermMemory(max_turns=10)
        mem.add_turn("user", "Hello")
        mem.add_turn("assistant", "Hi there")
        context = mem.get_context()
        assert "Hello" in context
        assert "Hi there" in context
    
    def test_clear(self):
        mem = ShortTermMemory()
        mem.add_turn("user", "Test")
        mem.clear()
        assert mem.turn_count == 0
    
    def test_entity_extraction(self):
        mem = ShortTermMemory()
        mem.add_turn("user", "Parle-moi de Vauxhall et RTX 5060")
        entities = mem.entities
        assert "Vauxhall" in entities or "RTX" in entities


# ─── Priority filter ─────────────────────────────────────────────────────────

class TestPriorityFilter:
    
    def test_correction_high_score(self):
        f = PriorityFilter()
        event = MemoryEvent(text="C'est faux, mon nom c'est Ali", source="user", category="correction")
        assert f.score(event) >= 0.3
    
    def test_preference_high_score(self):
        f = PriorityFilter()
        event = MemoryEvent(text="Je préfère les réponses courtes", source="user", category="user_preference")
        assert f.score(event) >= 0.3
    
    def test_greeting_low_score(self):
        f = PriorityFilter()
        event = MemoryEvent(text="Salut ça va", source="user", category="event")
        assert f.score(event) < 0.3
    
    def test_explicit_remember(self):
        f = PriorityFilter()
        event = MemoryEvent(text="Retiens bien que j'utilise Ubuntu", source="user")
        assert f.score(event) >= 0.3
    
    def test_should_store(self):
        f = PriorityFilter(threshold=0.3)
        important = MemoryEvent(text="Rappelle-toi je suis à Tourcoing", source="user", category="user_preference")
        trivial = MemoryEvent(text="ok super merci", source="user", category="event")
        assert f.should_store(important)
        assert not f.should_store(trivial)
    
    def test_decision_moderate_score(self):
        f = PriorityFilter()
        event = MemoryEvent(text="J'ai décidé de ne plus chercher de business en ligne", source="user")
        assert f.score(event) >= 0.3


# ─── Long-term memory (SQLite only — no Chroma in tests) ─────────────────────

class TestLongTermMemory:
    
    def test_store_and_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_memory.db")
            chroma = os.path.join(tmpdir, "test_chroma")
            store = LongTermMemory(db_path=db, chroma_path=chroma)
            
            event = MemoryEvent(text="Test memory", source="user")
            store.store(event)
            
            assert store.count() == 1
    
    def test_query_sqlite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_memory.db")
            chroma = os.path.join(tmpdir, "test_chroma")
            store = LongTermMemory(db_path=db, chroma_path=chroma)
            
            store.store(MemoryEvent(text="J'aime le Python", source="user", category="user_preference"))
            store.store(MemoryEvent(text="Le projet utilise faster-whisper", source="user", category="technical_knowledge"))
            
            hits = store.query("Python", limit=5)
            assert len(hits) >= 1
            assert any("Python" in h.text for h in hits)
    
    def test_query_category_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_memory.db")
            chroma = os.path.join(tmpdir, "test_chroma")
            store = LongTermMemory(db_path=db, chroma_path=chroma)
            
            store.store(MemoryEvent(text="Réponses courtes", source="user", category="user_preference"))
            store.store(MemoryEvent(text="RTX 5060", source="user", category="technical_knowledge"))
            
            hits = store.query("test", limit=5, category="user_preference")
            for h in hits:
                assert h.category == "user_preference"
    
    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_memory.db")
            chroma = os.path.join(tmpdir, "test_chroma")
            store = LongTermMemory(db_path=db, chroma_path=chroma)
            
            event = MemoryEvent(text="To delete", source="user")
            doc_id = store.store(event)
            assert store.count() == 1
            
            store.delete_by_id(doc_id)
            assert store.count() == 0
    
    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_memory.db")
            chroma = os.path.join(tmpdir, "test_chroma")
            store = LongTermMemory(db_path=db, chroma_path=chroma)
            
            store.store(MemoryEvent(text="One", source="user"))
            store.store(MemoryEvent(text="Two", source="user"))
            store.clear()
            assert store.count() == 0


# ─── Memory Manager (integration) ─────────────────────────────────────────────

class TestMemoryManager:
    
    def test_process_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_memory.db")
            chroma = os.path.join(tmpdir, "test_chroma")
            mgr = MemoryManager(db_path=db, chroma_path=chroma, filter_threshold=0.3)
            
            mgr.process(MemoryEvent(text="Salut", source="user"))
            assert mgr.short_term.turn_count == 1
    
    def test_important_events_stored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_memory.db")
            chroma = os.path.join(tmpdir, "test_chroma")
            mgr = MemoryManager(db_path=db, chroma_path=chroma, filter_threshold=0.3)
            
            mgr.process(MemoryEvent(
                text="Rappelle-toi mon nom c'est Ali",
                source="user",
                category="user_preference"
            ))
            assert mgr.long_term.count() >= 1
    
    def test_build_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_memory.db")
            chroma = os.path.join(tmpdir, "test_chroma")
            mgr = MemoryManager(db_path=db, chroma_path=chroma)
            
            mgr.process(MemoryEvent(text="Bonjour", source="user"))
            mgr.process(MemoryEvent(text="Salut !", source="assistant"))
            
            context = mgr.build_context("hello")
            assert "Bonjour" in context
    
    def test_record_preference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_memory.db")
            chroma = os.path.join(tmpdir, "test_chroma")
            mgr = MemoryManager(db_path=db, chroma_path=chroma, filter_threshold=0.1)
            
            mgr.record_preference("Je préfère le français")
            assert mgr.short_term.turn_count >= 1
    
    def test_new_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_memory.db")
            chroma = os.path.join(tmpdir, "test_chroma")
            mgr = MemoryManager(db_path=db, chroma_path=chroma)
            
            mgr.process(MemoryEvent(text="Old session", source="user"))
            mgr.new_session()
            assert mgr.short_term.turn_count == 0
    
    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = os.path.join(tmpdir, "test_memory.db")
            chroma = os.path.join(tmpdir, "test_chroma")
            mgr = MemoryManager(db_path=db, chroma_path=chroma)
            
            stats = mgr.get_stats()
            assert "short_term_turns" in stats
            assert "long_term_count" in stats
            assert "session_duration" in stats
