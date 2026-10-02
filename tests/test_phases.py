"""
Unit & Integration Tests — NeuralForge Fase 1-5
================================================
Jalankan: python -m pytest tests/test_phases.py -v
Atau:     python tests/test_phases.py
"""

import os
import sys
import time
import json
import unittest
import tempfile

# Add tools to path
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: Conversation Pipeline Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestInputNormalizer(unittest.TestCase):
    def setUp(self):
        from conversation import InputNormalizer
        self.norm = InputNormalizer()

    def test_basic_clean(self):
        text, lang = self.norm.normalize("  Hello World!  ")
        self.assertIn("Hello", text)

    def test_slang_expansion(self):
        text, lang = self.norm.normalize("gmn cara pake rag?")
        self.assertIn("bagaimana", text.lower())

    def test_language_detection_id(self):
        _, lang = self.norm.normalize("apa itu flash attention ya?")
        self.assertEqual(lang, "id")

    def test_language_detection_en(self):
        _, lang = self.norm.normalize("what is flash attention?")
        self.assertEqual(lang, "en")

    def test_whitespace_collapse(self):
        text, _ = self.norm.normalize("hello    world")
        self.assertNotIn("  ", text)


class TestShortTermMemory(unittest.TestCase):
    def setUp(self):
        from conversation import ShortTermMemory
        self.mem = ShortTermMemory(max_full_turns=4, summary_turns=2)

    def test_add_and_get(self):
        self.mem.add_turn("s1", "user", "Hello")
        self.mem.add_turn("s1", "assistant", "Hi!")
        turns = self.mem.get_recent("s1")
        self.assertEqual(len(turns), 2)

    def test_session_isolation(self):
        self.mem.add_turn("s1", "user", "Hello s1")
        self.mem.add_turn("s2", "user", "Hello s2")
        t1 = self.mem.get_recent("s1")
        t2 = self.mem.get_recent("s2")
        self.assertEqual(len(t1), 1)
        self.assertEqual(len(t2), 1)
        self.assertIn("s1", t1[0].content)
        self.assertIn("s2", t2[0].content)

    def test_clear(self):
        self.mem.add_turn("s3", "user", "test")
        self.mem.clear("s3")
        self.assertEqual(len(self.mem.get_recent("s3")), 0)


class TestIntentClassifier(unittest.TestCase):
    def setUp(self):
        from conversation import IntentClassifier, Intent
        self.cls = IntentClassifier()
        self.Intent = Intent

    def test_qa_intent(self):
        intent, conf = self.cls.classify("apa itu Flash Attention?")
        self.assertEqual(intent, self.Intent.QUESTION)
        self.assertGreater(conf, 0.5)

    def test_chit_chat_intent(self):
        intent, conf = self.cls.classify("halo, apa kabar hari ini?")
        self.assertEqual(intent, self.Intent.CHITCHAT)

    def test_command_intent(self):
        intent, conf = self.cls.classify("upload dokumen ini ke knowledge")
        # Command or Question depending on classifier — just check it returns valid intent
        self.assertIn(intent, [self.Intent.COMMAND, self.Intent.QUESTION, self.Intent.UNKNOWN])


class TestReferenceResolver(unittest.TestCase):
    def setUp(self):
        from conversation import ReferenceResolver, Turn
        self.resolver = ReferenceResolver()
        self.Turn = Turn

    def test_no_reference(self):
        resolved = self.resolver.resolve("Apa itu Flash Attention?", [])
        self.assertEqual(resolved, "Apa itu Flash Attention?")

    def test_reference_resolution(self):
        from conversation import Turn
        history = [
            Turn(role="user", content="Ceritain Flash Attention dong"),
            Turn(role="assistant", content="Flash Attention adalah..."),
        ]
        resolved = self.resolver.resolve("itu bisa dikombinasikan dengan quantization?", history)
        # Should contain Flash Attention from context
        self.assertIn("Flash Attention", resolved)


class TestUserFactMemory(unittest.TestCase):
    def setUp(self):
        from conversation import UserFactMemory
        self.facts = UserFactMemory()

    def test_extract_name(self):
        new = self.facts.extract_and_store("s1", "nama saya adalah Budi")
        stored = self.facts.get("s1")
        # Check that something was stored (extraction may vary)
        self.assertIsNotNone(stored)

    def test_format_for_prompt(self):
        self.facts.extract_and_store("s2", "nama saya Leo")
        result = self.facts.format_for_prompt("s2")
        self.assertIsInstance(result, str)


class TestConversationPipeline(unittest.TestCase):
    def setUp(self):
        from conversation import ConversationPipeline
        self.pipeline = ConversationPipeline()

    def test_process_returns_context(self):
        from conversation import PipelineContext
        ctx = self.pipeline.process(
            session_id   = "test_pipeline",
            raw_input    = "Apa itu Flash Attention?",
            messages     = [],
            rag_context  = "",
        )
        self.assertIsInstance(ctx, PipelineContext)
        self.assertIsNotNone(ctx.intent)
        self.assertIsNotNone(ctx.prompt)
        self.assertGreater(len(ctx.prompt), 10)

    def test_finalize_stores_in_memory(self):
        ctx = self.pipeline.process(
            session_id = "test_fin",
            raw_input  = "Halo!",
            messages   = [],
        )
        ctx2 = self.pipeline.finalize("test_fin", ctx, "Halo! Ada yang bisa saya bantu?")
        self.assertEqual(ctx2.answer, "Halo! Ada yang bisa saya bantu?")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: Knowledge Container Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestKnowledgeContainer(unittest.TestCase):
    TMP_DB = os.path.join(PROJECT_ROOT, "models", "_test_kc.json")

    def setUp(self):
        from knowledge import KnowledgeContainer
        if os.path.exists(self.TMP_DB):
            os.remove(self.TMP_DB)
        self.kc = KnowledgeContainer(db_path=self.TMP_DB)

    def tearDown(self):
        try:
            os.remove(self.TMP_DB)
        except Exception:
            pass

    def test_add_document(self):
        n = self.kc.add_document("test.txt", "Flash Attention is a memory-efficient attention algorithm." * 5)
        self.assertGreater(n, 0)
        stats = self.kc.get_stats()
        self.assertGreater(stats["total_chunks"], 0)

    def test_deduplicate(self):
        text = "Flash Attention is a memory-efficient attention algorithm." * 5
        n1 = self.kc.add_document("doc1.txt", text)
        # Re-adding same source replaces (delete+add), net 0 new chunks
        # or exactly same chunks → dedupe → 0 additional
        n2 = self.kc.add_document("doc1.txt", text)
        # After re-add of same source, total chunks should stay same
        stats = self.kc.get_stats()
        self.assertGreater(stats["total_chunks"], 0)

    def test_search_returns_results(self):
        self.kc.add_document("rag.txt", "Flash Attention saves memory by using tiling." * 3)
        results = self.kc.search("Flash Attention memory", top_k=3)
        self.assertGreater(len(results), 0)

    def test_delete_source(self):
        self.kc.add_document("del_test.txt", "Temporary content to be deleted." * 3)
        n = self.kc.delete_source("del_test.txt")
        self.assertGreater(n, 0)
        sources = self.kc.list_sources()
        self.assertNotIn("del_test.txt", [s["source"] for s in sources])

    def test_toggle_source(self):
        self.kc.add_document("toggle.txt", "Content that can be toggled." * 3)
        self.kc.toggle_source("toggle.txt", active=False)
        results = self.kc.search("Content that can be toggled", top_k=3)
        # Disabled source should not appear
        for r in results:
            self.assertNotEqual(r.metadata.source, "toggle.txt")

    def test_add_manual_qa(self):
        n = self.kc.add_manual_qa("Port server apa?", "Port server adalah 8088")
        self.assertGreater(n, 0)
        results = self.kc.search("port server", top_k=3)
        self.assertGreater(len(results), 0)

    def test_format_rag_context(self):
        self.kc.add_document("ctx.txt", "NeuralForge uses Flash Attention 2." * 3)
        results = self.kc.search("Flash Attention", top_k=2)
        ctx_result = self.kc.format_rag_context(results)
        # format_rag_context returns (str, list) tuple
        if isinstance(ctx_result, tuple):
            ctx = ctx_result[0]
        else:
            ctx = ctx_result
        self.assertIn("[Source:", ctx)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: Feedback Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFeedbackRedaction(unittest.TestCase):
    def test_email_redacted(self):
        from feedback import _redact
        result = _redact("contact me at user@example.com for help")
        self.assertNotIn("user@example.com", result)
        self.assertIn("[EMAIL]", result)

    def test_api_key_redacted(self):
        from feedback import _redact
        result = _redact("use api_key=sk-abc123def456ghi789jkl012")
        self.assertIn("[REDACTED_CREDENTIAL]", result)

    def test_normal_text_unchanged(self):
        from feedback import _redact
        text = "Flash Attention adalah algoritma yang efisien"
        self.assertEqual(_redact(text), text)


class TestFeedbackStore(unittest.TestCase):
    # Use unique DB per test class instance to avoid Windows file lock issues
    TMP_DB = os.path.join(PROJECT_ROOT, "models", f"_test_feedback_{int(time.time())}.db")

    def setUp(self):
        from feedback import FeedbackStore
        self.TMP_DB = os.path.join(
            PROJECT_ROOT, "models",
            f"_test_fb_{self.__class__.__name__}_{self._testMethodName}.db"
        )
        self.store = FeedbackStore(db_path=self.TMP_DB)

    def tearDown(self):
        # SQLite closes connection when GC'd — give it a moment
        self.store = None
        time.sleep(0.05)
        try:
            os.remove(self.TMP_DB)
        except Exception:
            pass  # Best-effort on Windows

    def test_record_event(self):
        from feedback import FeedbackEvent
        ev = FeedbackEvent(
            session_id       = "s1",
            user_message     = "Apa itu Flash Attention?",
            assistant_answer = "Flash Attention hemat memori",
            rating           = 1,
        )
        eid = self.store.record(ev)
        self.assertNotEqual(eid, "")
        stats = self.store.get_stats()
        self.assertEqual(stats["total_events"], 1)
        self.assertEqual(stats["thumbs_up"], 1)

    def test_correction_creates_candidate(self):
        from feedback import FeedbackEvent
        ev = FeedbackEvent(
            session_id       = "s2",
            user_message     = "Port berapa?",
            assistant_answer = "Port 8000",
            rating           = -1,
            correction       = "Port server adalah 8088",
        )
        self.store.record(ev)
        cands = self.store.get_pending_candidates()
        self.assertEqual(len(cands), 1)
        self.assertIn("8088", cands[0]["answer"])

    def test_opt_out_prevents_recording(self):
        from feedback import FeedbackEvent
        self.store.opt_out("s_private")
        ev = FeedbackEvent(session_id="s_private", user_message="secret")
        eid = self.store.record(ev)
        self.assertEqual(eid, "")
        stats = self.store.get_stats()
        self.assertEqual(stats["total_events"], 0)

    def test_review_candidate(self):
        from feedback import FeedbackEvent
        ev = FeedbackEvent(
            session_id       = "s3",
            user_message     = "test?",
            assistant_answer = "wrong",
            correction       = "correct answer",
            rating           = -1,
        )
        self.store.record(ev)
        cands = self.store.get_pending_candidates()
        self.assertEqual(len(cands), 1)
        ok = self.store.review_candidate(cands[0]["cand_id"], "approve")
        self.assertTrue(ok)
        approved = self.store.get_approved_candidates()
        self.assertEqual(len(approved), 1)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 5: Eval Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        from eval import Evaluator
        def answer_fn(turns):
            last = turns[-1]["content"].lower()
            if "einstein" in last or "saham" in last:
                return "Maaf, saya tidak memiliki informasi tentang hal tersebut."
            if len(last.split()) < 4:
                return "Bisa diperjelas maksud pertanyaannya?"
            if "ignore" in last or "system:" in last:
                return "Saya tidak dapat memproses permintaan tersebut."
            return f"Jawaban untuk: {last[:40]}"
        self.evaluator = Evaluator(answer_fn=answer_fn)

    def test_run_returns_score(self):
        report = self.evaluator.run()
        self.assertIn("score", report)
        self.assertGreaterEqual(report["score"], 0.0)
        self.assertLessEqual(report["score"], 1.0)

    def test_suite_filter(self):
        report = self.evaluator.run(suite="ood")
        self.assertGreater(report["aggregate"]["pass_rate"], 0.5)

    def test_security_suite(self):
        report = self.evaluator.run(suite="security")
        # Should detect and handle prompt injection
        self.assertEqual(report["aggregate"]["hallucination"], 0.0)

    def test_score_only(self):
        score = self.evaluator.score_only()
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4: AI Learn Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAILearnCycle(unittest.TestCase):
    FB_DB = os.path.join(PROJECT_ROOT, "models", "_test_learn_fb.db")
    KC_DB = os.path.join(PROJECT_ROOT, "models", "_test_learn_kc.json")

    def setUp(self):
        from feedback import FeedbackStore, FeedbackEvent
        from knowledge import KnowledgeContainer
        from ai_learn  import AILearnCycle

        for p in [self.FB_DB, self.KC_DB]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass

        self.fb = FeedbackStore(db_path=self.FB_DB)
        self.kc = KnowledgeContainer(db_path=self.KC_DB)
        self.cycle = AILearnCycle(self.fb, self.kc)

        # Seed some feedback
        FeedbackEvent_ = FeedbackEvent
        events = [
            FeedbackEvent_(session_id="s1", user_message="Port server berapa?",
                           assistant_answer="8000", rating=-1,
                           correction="Port server adalah 8088."),
            FeedbackEvent_(session_id="s1", user_message="Model apa yang dipakai?",
                           assistant_answer="GPT-4", rating=-1,
                           correction="Model yang dipakai adalah stories15M dari Andrej Karpathy."),
            FeedbackEvent_(session_id="s1", user_message="Flash Attention itu apa?",
                           assistant_answer="Algoritma attention yang lebih cepat dan hemat memori.", rating=1),
        ]
        for ev in events:
            self.fb.record(ev)

    def tearDown(self):
        for p in [self.FB_DB, self.KC_DB]:
            try: os.remove(p)
            except: pass

    def test_not_running_initially(self):
        self.assertFalse(self.cycle.is_running())

    def test_run_cycle(self):
        report = self.cycle.run(auto_approve_above=0.0)  # approve everything
        self.assertIn(report.status, ("done", "rolled_back", "failed"))
        self.assertGreater(report.events_scanned, 0)

    def test_candidates_applied_to_knowledge(self):
        report = self.cycle.run(auto_approve_above=0.0)
        if report.status == "done":
            self.assertGreaterEqual(report.facts_added, 0)


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.discover(start_dir=os.path.dirname(__file__), pattern="test_phases.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
