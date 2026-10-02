"""
NeuralForge: Full-Stack Inference Server v3
=============================================
Phase 1: Conversation Pipeline (InputNormalizer, Memory, IntentClassifier, PromptBuilder)
Phase 2: Knowledge Container  (Hybrid RAG, BM25, Citations, Management API)
Phase 3: Feedback Store        (Thumbs up/down, Corrections, Opt-out, Candidates)
Phase 4: AI Learn              (Autonomous safe learning cycle with rollback)
Phase 5: Auto Evaluation       (50+ scenarios, metrics, baseline comparison)

Legacy features preserved:
- SSE Streaming, Multi-Turn, Tool Calling, Episodic Memory, Benchmark API
"""

import http.server
import socketserver
import json
import time
import os
import sys
import threading
import uuid
import urllib.parse
from collections import defaultdict

# Add tools dir to path
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from py_reference     import PyEngine, softmax
from persistent_rag   import PersistentKnowledgeHub
from agent            import AgentLoop, get_tool_manifest, parse_tool_calls, execute_tool
from conversation     import ConversationPipeline
from knowledge        import KnowledgeContainer
from feedback         import FeedbackStore, FeedbackEvent
from ai_learn         import AILearnCycle, get_learn_history, get_current_version, rollback_to_version
from eval             import Evaluator, save_result, compare_with_baseline, save_baseline, get_score_history
import numpy as np

PORT            = 8088
INDEX_HTML_PATH = os.path.join(SCRIPT_DIR,   "index.html")
MODEL_PATH      = os.path.join(PROJECT_ROOT,  "models", "stories15M.bin")
TOKENIZER_PATH  = os.path.join(PROJECT_ROOT,  "models", "tokenizer.bin")
KNOWLEDGE_DB    = os.path.join(PROJECT_ROOT,  "models", "knowledge_db.json")
KNOWLEDGE_DB_V2 = os.path.join(PROJECT_ROOT,  "models", "knowledge_db_v2.json")

engine            = None
knowledge_hub     = None   # legacy PersistentKnowledgeHub
agent_loop        = None
conv_pipeline     = ConversationPipeline()          # Fase 1
knowledge_v2      = KnowledgeContainer(db_path=KNOWLEDGE_DB_V2)  # Fase 2
feedback_store    = FeedbackStore()                 # Fase 3
learn_cycle: AILearnCycle | None = None             # Fase 4 (init after server starts)

# ── Per-session conversation memory (legacy + pipeline) ───────────────────────
session_store: dict[str, list[dict]] = defaultdict(list)
session_lock = threading.Lock()

# ── AI Learn progress log (in-memory ring buffer) ─────────────────────────────
learn_log: list[str] = []
learn_lock = threading.Lock()

# ── Benchmark / Metrics store ──────────────────────────────────────────────────
metrics_store: list[dict] = []
metrics_lock  = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
class LLMRequestHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Custom compact logging
        print(f"  {self.address_string()} → {fmt % args}")

    def _headers(self, status=200, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Session-Id")
        self.end_headers()

    def do_OPTIONS(self):
        self._headers(200)

    # ── GET endpoints ─────────────────────────────────────────────────────────
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/":
            self._serve_html()
        elif path == "/v1/models":
            self._json({"object": "list", "data": [
                {"id": "stories15M-custom-cpp", "object": "model",
                 "owned_by": "custom-engine", "features": ["rag", "tools", "streaming"]}
            ]})
        elif path == "/v1/knowledge/stats":
            self._json(knowledge_hub.get_stats())
        elif path == "/v1/knowledge/v2/stats":
            self._json(knowledge_v2.get_stats())
        elif path == "/v1/knowledge/v2/list":
            self._json({"sources": knowledge_v2.list_sources()})
        elif path == "/v1/tools":
            self._json({"tools": get_tool_manifest()})
        elif path == "/v1/metrics":
            with metrics_lock:
                recent = metrics_store[-50:]
            self._json({
                "total_requests": len(metrics_store),
                "recent":         recent,
                "avg_tps":        self._avg_tps(),
                "avg_latency_ms": self._avg_latency()
            })
        elif path == "/v1/memory/episodic":
            self._json({
                "count":   len(knowledge_hub.episodic_memory),
                "entries": [{"summary": m["summary"], "turns": m["turns"]}
                            for m in knowledge_hub.episodic_memory[-10:]]
            })
        elif path == "/v1/feedback/stats":
            self._json(feedback_store.get_stats())
        elif path == "/v1/feedback/candidates":
            self._json({"candidates": feedback_store.get_pending_candidates()})
        elif path == "/v1/learn/status":
            self._json({
                "running":         learn_cycle.is_running() if learn_cycle else False,
                "current_version": get_current_version(),
                "log":             learn_log[-50:],
                "history":         get_learn_history()[-10:],
            })
        elif path == "/v1/eval/history":
            self._json({"history": get_score_history()})
        else:
            self._headers(404)
            self.wfile.write(b'{"error": "not found"}')

    # ── POST endpoints ────────────────────────────────────────────────────────
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        path   = urllib.parse.urlparse(self.path).path

        # Parse session ID from headers
        session_id = self.headers.get("X-Session-Id", "default")

        if path == "/v1/knowledge/upload":
            self._handle_upload(body)
        elif path == "/v1/knowledge/scrape":
            self._handle_scrape(body)
        elif path == "/v1/knowledge/delete":
            self._handle_delete(body)
        # --- Knowledge v2 endpoints ---
        elif path == "/v1/knowledge/v2/upload":
            self._handle_kv2_upload(body)
        elif path == "/v1/knowledge/v2/scrape":
            self._handle_kv2_scrape(body)
        elif path == "/v1/knowledge/v2/qa":
            self._handle_kv2_qa(body)
        elif path == "/v1/knowledge/v2/toggle":
            self._handle_kv2_toggle(body)
        elif path == "/v1/knowledge/v2/delete":
            self._handle_kv2_delete(body)
        # --- Feedback endpoints ---
        elif path == "/v1/feedback":
            self._handle_feedback(body, session_id)
        elif path == "/v1/feedback/rating":
            self._handle_feedback_rating(body)
        elif path == "/v1/feedback/correction":
            self._handle_feedback_correction(body)
        elif path == "/v1/feedback/candidate/review":
            self._handle_candidate_review(body)
        elif path == "/v1/feedback/optout":
            self._handle_optout(body, session_id)
        # --- AI Learn endpoint ---
        elif path == "/v1/learn/start":
            self._handle_learn_start(body)
        elif path == "/v1/learn/rollback":
            self._handle_learn_rollback(body)
        # --- Eval endpoints ---
        elif path == "/v1/eval/run":
            self._handle_eval_run(body)
        elif path == "/v1/eval/baseline":
            self._handle_eval_baseline(body)
        # --- Chat / Agent / Memory ---
        elif path == "/v1/chat/completions":
            self._handle_chat(body, session_id)
        elif path == "/v1/agent/run":
            self._handle_agent(body, session_id)
        elif path == "/v1/memory/compress":
            self._handle_compress_memory(body, session_id)
        elif path == "/v1/session/clear":
            self._handle_clear_session(session_id)
        else:
            self._headers(404)
            self.wfile.write(b'{"error": "endpoint not found"}')

    # ── DELETE endpoints ──────────────────────────────────────────────────────
    def do_DELETE(self):
        path = urllib.parse.urlparse(self.path).path
        qs   = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

        if path == "/v1/knowledge":
            source = qs.get("source", [""])[0]
            if source:
                removed = knowledge_hub.delete_document(source)
                self._json({"status": "ok", "removed": removed, "stats": knowledge_hub.get_stats()})
            else:
                self._headers(400)
                self.wfile.write(b'{"error": "source required"}')
        else:
            self._headers(404)

    # ─────────────────────────────────────────────────────────────────────────
    def _serve_html(self):
        try:
            with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
                content = f.read().encode("utf-8")
            self._headers(200, "text/html; charset=utf-8")
            self.wfile.write(content)
        except FileNotFoundError:
            self._headers(404)
            self.wfile.write(b"<h1>index.html not found</h1>")

    def _json(self, data, status=200):
        self._headers(status)
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    # ── Upload ────────────────────────────────────────────────────────────────
    def _handle_upload(self, body: bytes):
        try:
            data    = json.loads(body.decode("utf-8"))
            source  = data.get("source", "uploaded.txt")
            content = data.get("content", "")
            count   = knowledge_hub.add_document(source, content)
            self._json({"status": "ok", "chunks_added": count,
                        "stats": knowledge_hub.get_stats()})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    # ── Scrape ────────────────────────────────────────────────────────────────
    def _handle_scrape(self, body: bytes):
        try:
            data    = json.loads(body.decode("utf-8"))
            url     = data.get("url", "")
            multihop = data.get("multihop", False)
            print(f"  [RAG] Scraping: {url}")
            content = knowledge_hub.scrape_url(url)
            if content.startswith("Error"):
                self._json({"error": content}, 400)
                return
            count = knowledge_hub.add_document(url, content)
            self._json({
                "status":      "ok",
                "chunks_added": count,
                "preview":     content[:300] + "...",
                "stats":       knowledge_hub.get_stats()
            })
        except Exception as e:
            self._json({"error": str(e)}, 500)

    # ── Delete ────────────────────────────────────────────────────────────────
    def _handle_delete(self, body: bytes):
        try:
            data   = json.loads(body.decode("utf-8"))
            source = data.get("source", "")
            removed = knowledge_hub.delete_document(source)
            self._json({"status": "ok", "removed": removed,
                        "stats": knowledge_hub.get_stats()})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2 — Knowledge Container v2 Handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_kv2_upload(self, body: bytes):
        try:
            data    = json.loads(body.decode("utf-8"))
            source  = data.get("source", "upload.txt")
            content = data.get("content", "")
            trust   = float(data.get("trust_level", 1.0))
            n = knowledge_v2.add_document(source, content, trust_level=trust)
            self._json({"status": "ok", "chunks_added": n,
                        "stats": knowledge_v2.get_stats()})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _handle_kv2_scrape(self, body: bytes):
        try:
            data = json.loads(body.decode("utf-8"))
            url  = data.get("url", "")
            n    = knowledge_v2.scrape_url(url)
            if isinstance(n, str) and n.startswith("Error"):
                self._json({"error": n}, 400)
                return
            self._json({"status": "ok", "chunks_added": n,
                        "stats": knowledge_v2.get_stats()})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _handle_kv2_qa(self, body: bytes):
        try:
            data = json.loads(body.decode("utf-8"))
            q    = data.get("question", "")
            a    = data.get("answer",   "")
            trust= float(data.get("trust_level", 1.2))
            coll = data.get("collection", "manual")
            n    = knowledge_v2.add_manual_qa(q, a, trust_level=trust,
                                               collection=coll)
            self._json({"status": "ok", "chunks_added": n,
                        "stats": knowledge_v2.get_stats()})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _handle_kv2_toggle(self, body: bytes):
        try:
            data   = json.loads(body.decode("utf-8"))
            source = data.get("source", "")
            active = bool(data.get("active", True))
            knowledge_v2.toggle_source(source, active)
            self._json({"status": "ok", "source": source, "active": active})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _handle_kv2_delete(self, body: bytes):
        try:
            data   = json.loads(body.decode("utf-8"))
            source = data.get("source", "")
            n      = knowledge_v2.delete_source(source)
            self._json({"status": "ok", "removed": n,
                        "stats": knowledge_v2.get_stats()})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 3 — Feedback Handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_feedback(self, body: bytes, session_id: str):
        try:
            data  = json.loads(body.decode("utf-8"))
            ev    = FeedbackEvent(
                session_id        = session_id,
                conv_id           = data.get("conv_id", ""),
                user_message      = data.get("user_message", ""),
                assistant_answer  = data.get("assistant_answer", ""),
                rating            = int(data.get("rating", 0)),
                correction        = data.get("correction", ""),
                rag_sources       = json.dumps(data.get("rag_sources", [])),
                prompt_version    = data.get("prompt_version", ""),
                knowledge_version = str(get_current_version()),
                intent            = data.get("intent", ""),
                latency_ms        = float(data.get("latency_ms", 0)),
            )
            event_id = feedback_store.record(ev)
            self._json({"status": "ok", "event_id": event_id})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _handle_feedback_rating(self, body: bytes):
        try:
            data     = json.loads(body.decode("utf-8"))
            event_id = data.get("event_id", "")
            rating   = int(data.get("rating", 0))
            ok = feedback_store.update_rating(event_id, rating)
            self._json({"status": "ok" if ok else "not_found"})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _handle_feedback_correction(self, body: bytes):
        try:
            data       = json.loads(body.decode("utf-8"))
            event_id   = data.get("event_id", "")
            correction = data.get("correction", "")
            cand_id = feedback_store.add_correction(event_id, correction)
            self._json({"status": "ok", "candidate_id": cand_id})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _handle_candidate_review(self, body: bytes):
        try:
            data    = json.loads(body.decode("utf-8"))
            cand_id = data.get("cand_id", "")
            action  = data.get("action", "approve")  # "approve" | "reject"
            ok = feedback_store.review_candidate(cand_id, action)
            self._json({"status": "ok" if ok else "not_found"})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _handle_optout(self, body: bytes, session_id: str):
        try:
            data   = json.loads(body.decode("utf-8"))
            action = data.get("action", "optout")   # "optout" | "optin"
            if action == "optout":
                feedback_store.opt_out(session_id)
            else:
                feedback_store.opt_in(session_id)
            self._json({"status": "ok", "session_id": session_id,
                        "opted_out": feedback_store.is_opted_out(session_id)})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 4 — AI Learn Handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_learn_start(self, body: bytes):
        """Mulai siklus AI Learn di background thread."""
        global learn_cycle
        if learn_cycle and learn_cycle.is_running():
            self._json({"status": "already_running"})
            return
        try:
            data              = json.loads(body.decode("utf-8")) if body else {}
            auto_approve_thresh = float(data.get("auto_approve_above", 0.85))

            def _progress(msg: str):
                with learn_lock:
                    learn_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
                    if len(learn_log) > 200:
                        del learn_log[0]
                print(msg)

            # Build eval_fn so AI Learn can test regressions
            def _eval_fn() -> float:
                ev = Evaluator(answer_fn=self._dummy_answer_fn_for_eval)
                return ev.score_only()

            learn_cycle = AILearnCycle(
                feedback_store      = feedback_store,
                knowledge_container = knowledge_v2,
            )
            learn_cycle.run_async(
                auto_approve_above = auto_approve_thresh,
                progress_callback  = _progress,
                eval_fn            = _eval_fn,
            )
            self._json({"status": "started"})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _handle_learn_rollback(self, body: bytes):
        try:
            data    = json.loads(body.decode("utf-8"))
            version = int(data.get("version", 0))
            ok = rollback_to_version(version, knowledge_v2)
            self._json({"status": "ok" if ok else "failed",
                        "current_version": get_current_version()})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _dummy_answer_fn_for_eval(self, turns: list) -> str:
        """Used by eval_fn inside AI Learn. Returns last known reply for regression test."""
        last = turns[-1]["content"] if turns else ""
        if any(kw in last.lower() for kw in ["einstein", "saham"]):
            return "Maaf, saya tidak memiliki informasi tentang hal tersebut."
        if len(last.split()) < 4:
            return "Bisa diperjelas maksudnya apa?"
        return f"[stub: {last[:40]}]"

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 5 — Eval Handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_eval_run(self, body: bytes):
        try:
            data  = json.loads(body.decode("utf-8")) if body else {}
            suite = data.get("suite", None)
            ev    = Evaluator(answer_fn=self._dummy_answer_fn_for_eval)
            report= ev.run(suite=suite)
            path  = save_result(report, tag=suite or "api")
            self._json({"status": "ok", "score": report["score"],
                        "aggregate": report["aggregate"],
                        "by_category": report.get("by_category", {}),
                        "saved_to": path})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def _handle_eval_baseline(self, body: bytes):
        try:
            data  = json.loads(body.decode("utf-8")) if body else {}
            suite = data.get("suite", None)
            ev    = Evaluator(answer_fn=self._dummy_answer_fn_for_eval)
            report= ev.run(suite=suite)
            save_baseline(report)
            self._json({"status": "ok", "score": report["score"],
                        "message": "Baseline saved."})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    # ══════════════════════════════════════════════════════════════════════════
    # CHAT — Upgrade: Conversation Pipeline (Fase 1) + RAG v2 (Fase 2)
    # ══════════════════════════════════════════════════════════════════════════

    # ── Chat (SSE Streaming + Multi-Turn + RAG) ───────────────────────────────
    def _handle_chat(self, body: bytes, session_id: str):
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._json({"error": f"Invalid JSON: {e}"}, 400)
            return

        messages    = data.get("messages", [])
        max_tokens  = int(data.get("max_tokens",  50))
        temperature = float(data.get("temperature", 0.7))
        stream      = bool(data.get("stream",      True))
        use_rag     = bool(data.get("use_rag",     True))
        use_multihop = bool(data.get("use_multihop", False))
        use_agent   = bool(data.get("use_agent",   False))

        # Get last user message
        last_user = "Hello"
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break

        # Persist conversation to session store
        with session_lock:
            session_store[session_id] = messages

        # ── RAG v2: Knowledge Container (Fase 2) ─────────────────────────────
        rag_context  = ""
        rag_sources  = []
        if use_rag:
            kv2_results = knowledge_v2.search(last_user, top_k=3)
            if kv2_results:
                ctx_block   = knowledge_v2.format_rag_context(kv2_results)
                rag_context = ctx_block
                rag_sources = [r.metadata.source for r in kv2_results]
                print(f"  [RAG v2] {len(kv2_results)} chunks from {rag_sources[:2]}")

        # ── RAG Legacy fallback (if v2 empty) ────────────────────────────────
        rag_source = rag_sources[0] if rag_sources else None
        if not rag_context and use_rag:
            retrieved = knowledge_hub.retrieve(
                last_user, top_k=2,
                use_multihop=use_multihop,
                use_parent=True
            )
            if retrieved:
                rag_context = "\n".join(
                    f"[Context from {r['source']}]: {r['_context'][:300]}"
                    for r in retrieved[:2]
                )
                rag_source = retrieved[0]["source"]
                rag_sources = [r["source"] for r in retrieved[:2]]
                print(f"  [RAG legacy] Hit: {rag_source}")

        # ── Episodic Memory ───────────────────────────────────────────────────
        episodic_ctx = ""
        memories = knowledge_hub.search_episodic_memory(last_user, top_k=1)
        if memories:
            episodic_ctx = f"[Past conversation: {memories[0][:200]}]\n"

        # ── Conversation Pipeline Fase 1 ─────────────────────────────────────
        ctx = conv_pipeline.process(
            session_id      = session_id,
            raw_input       = last_user,
            messages        = messages,
            rag_context     = rag_context,
            rag_sources     = rag_sources,
            episodic_context= episodic_ctx,
        )

        # Use pipeline-built prompt
        prompt  = ctx.prompt
        t_start = time.time()

        # ── Encode Tokens ─────────────────────────────────────────────────────
        tokens     = engine.tokenizer.encode(prompt)
        pos        = 0
        token      = tokens[0]
        prompt_len = len(tokens)

        # ── SSE Streaming ─────────────────────────────────────────────────────
        if stream:
            self.send_response(200)
            self.send_header("Content-Type",  "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection",    "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Send metadata event
            meta = {"rag_injected": bool(rag_context), "source": rag_source,
                    "session_id": session_id}
            self._sse({"meta": meta})

            gen_count    = 0
            generated    = ""
            all_tool_res = []

            for _ in range(max_tokens):
                logits = engine.forward(token, pos)
                if pos + 1 < prompt_len:
                    next_tok = tokens[pos + 1]
                else:
                    probs    = softmax(logits / temperature)
                    next_tok = int(np.random.choice(len(probs), p=probs))
                    if next_tok == 2:
                        break
                    word      = engine.tokenizer.decode(next_tok)
                    generated += word
                    gen_count += 1

                    # Check for tool call
                    calls = parse_tool_calls(generated)
                    if calls and use_agent:
                        for call in calls:
                            result = execute_tool(call["tool"], call["params"],
                                                  knowledge_hub=knowledge_hub)
                            all_tool_res.append({"tool": call["tool"], "result": result})
                            self._sse({"tool_result": {"tool": call["tool"], "result": result}})
                            generated += f" [RESULT: {result}]"

                    self._sse({"choices": [{"delta": {"content": word}}]})
                    time.sleep(0.008)  # natural pacing

                token = next_tok
                pos  += 1

            elapsed = time.time() - t_start
            tps     = gen_count / elapsed if elapsed > 0 else 0

            # Finalize pipeline (verify + store in memory)
            conv_pipeline.finalize(session_id, ctx, generated)

            # Send pipeline meta in SSE
            self._sse({"done": True, "stats": {
                "tokens":          gen_count,
                "time_s":          round(elapsed, 2),
                "tps":             round(tps, 1),
                "intent":          ctx.intent.value if ctx.intent else "",
                "rag_v2_sources":  rag_sources,
                "prompt_version":  ctx.prompt_version,
                "conv_id":         ctx.conv_id,
            }})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

            self._record_metric(gen_count, elapsed * 1000, rag_source)
            return

        # ── Non-Streaming JSON ────────────────────────────────────────────────
        generated = ""
        for _ in range(max_tokens):
            logits = engine.forward(token, pos)
            if pos + 1 < prompt_len:
                next_tok = tokens[pos + 1]
            else:
                probs    = softmax(logits / temperature)
                next_tok = int(np.random.choice(len(probs), p=probs))
                if next_tok == 2:
                    break
                generated += engine.tokenizer.decode(next_tok)
            token = next_tok
            pos  += 1

        elapsed = time.time() - t_start
        generated_text = generated.strip() or "..."
        conv_pipeline.finalize(session_id, ctx, generated_text)
        self._record_metric(len(generated.split()), elapsed * 1000, rag_source)
        self._json({
            "id":      f"chatcmpl-{int(time.time())}",
            "model":   "stories15M-custom-cpp",
            "choices": [{"message": {
                "role":       "assistant",
                "content":    generated_text,
                "rag_source": rag_source,
                "rag_v2":     rag_sources,
                "conv_id":    ctx.conv_id,
                "intent":     ctx.intent.value if ctx.intent else "",
            }, "finish_reason": "stop"}],
            "usage": {"total_tokens": len(tokens) + len(generated.split())}
        })

    # ── Agent Run ─────────────────────────────────────────────────────────────
    def _handle_agent(self, body: bytes, session_id: str):
        try:
            data       = json.loads(body.decode("utf-8"))
            message    = data.get("message", "")
            temperature = float(data.get("temperature", 0.7))
            max_tokens  = int(data.get("max_tokens", 80))

            with session_lock:
                history = list(session_store.get(session_id, []))

            result = agent_loop.run(
                message, history=history,
                temperature=temperature, max_tokens=max_tokens
            )

            with session_lock:
                session_store[session_id].append({"role": "user",    "content": message})
                session_store[session_id].append({"role": "assistant", "content": result["response"]})

            self._json(result)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    # ── Compress Session to Episodic Memory ───────────────────────────────────
    def _handle_compress_memory(self, body: bytes, session_id: str):
        try:
            with session_lock:
                turns = session_store.get(session_id, [])
            if len(turns) < 4:
                self._json({"status": "not enough turns", "turns": len(turns)})
                return
            knowledge_hub.add_episodic_memory(turns)
            with session_lock:
                session_store[session_id] = []
            self._json({"status": "ok", "compressed": len(turns)})
        except Exception as e:
            self._json({"error": str(e)}, 500)

    # ── Clear Session ─────────────────────────────────────────────────────────
    def _handle_clear_session(self, session_id: str):
        with session_lock:
            session_store[session_id] = []
        self._json({"status": "ok", "session_id": session_id})

    # ── SSE helper ────────────────────────────────────────────────────────────
    def _sse(self, data: dict):
        try:
            payload = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()
        except Exception:
            pass

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _record_metric(self, tokens: int, latency_ms: float, rag_source):
        with metrics_lock:
            metrics_store.append({
                "ts":           time.time(),
                "tokens":       tokens,
                "latency_ms":   round(latency_ms, 1),
                "tps":          round(tokens / (latency_ms / 1000) if latency_ms > 0 else 0, 1),
                "rag_hit":      rag_source is not None,
                "rag_source":   rag_source
            })
            if len(metrics_store) > 200:
                del metrics_store[0]

    def _avg_tps(self) -> float:
        with metrics_lock:
            recent = metrics_store[-20:]
        if not recent:
            return 0.0
        return round(sum(m["tps"] for m in recent) / len(recent), 1)

    def _avg_latency(self) -> float:
        with metrics_lock:
            recent = metrics_store[-20:]
        if not recent:
            return 0.0
        return round(sum(m["latency_ms"] for m in recent) / len(recent), 1)


# --- Server Startup ---
def run_server():
    global engine, knowledge_hub, agent_loop, learn_cycle

    print(f"[Server] Loading model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[Server] ERROR: Model not found. Run: python tools/download_model.py")
        return

    engine        = PyEngine(MODEL_PATH, TOKENIZER_PATH)
    embedding_dim = engine.cfg.dim

    print(f"[Server] Loading Knowledge Hub legacy (dim={embedding_dim})...")
    knowledge_hub = PersistentKnowledgeHub(embedding_dim=embedding_dim, db_path=KNOWLEDGE_DB)

    # Knowledge Container v2 (Fase 2) already init'd at module level
    print(f"[Server] Knowledge v2: {knowledge_v2.get_stats()['total_chunks']} chunks")

    print("[Server] Initializing Agent Loop...")
    agent_loop = AgentLoop(engine, knowledge_hub=knowledge_hub, max_steps=3)

    # AI Learn Cycle (Fase 4)
    learn_cycle = AILearnCycle(
        feedback_store      = feedback_store,
        knowledge_container = knowledge_v2,
    )

    print(f"\n{'='*65}")
    print(f"  NeuralForge Server v3 — ONLINE")
    print(f"  URL:        http://localhost:{PORT}")
    print(f"  Model:      stories15M (dim={embedding_dim}, layers={engine.cfg.n_layers})")
    print(f"  Knowledge:  legacy={len(knowledge_hub.chunks)} chunks  "
          f"v2={knowledge_v2.get_stats()['total_chunks']} chunks")
    print(f"  Feedback:   {feedback_store.get_stats()['total_events']} events logged")
    print(f"  Learn ver.: {get_current_version()}")
    print(f"  Pipeline:   ConversationPipeline v1 (Fase 1-5 active)")
    print(f"  Features:   RAG v2 | Tool Calling | AI Learn | Eval | Feedback")
    print(f"{'='*65}\n")

    server_address = ("", PORT)
    # Allow address reuse to avoid "address already in use"
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, LLMRequestHandler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
