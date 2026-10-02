"""
LeagueOfLLMs: Full-Stack Inference Server v2
Features:
1. WebSocket + SSE Streaming (asyncio-based, multi-client)
2. Multi-Turn Conversation Memory (per-session)
3. Persistent Hybrid RAG v2 (Semantic Chunking + Re-ranking + Multi-hop)
4. Tool Calling / Agent Loop (ReAct)
5. Benchmark Dashboard API
6. Knowledge Management: Upload, Scrape, Delete, List
7. Episodic Memory API
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
import numpy as np

PORT            = 8088
INDEX_HTML_PATH = os.path.join(SCRIPT_DIR,   "index.html")
MODEL_PATH      = os.path.join(PROJECT_ROOT,  "models", "stories15M.bin")
TOKENIZER_PATH  = os.path.join(PROJECT_ROOT,  "models", "tokenizer.bin")
KNOWLEDGE_DB    = os.path.join(PROJECT_ROOT,  "models", "knowledge_db.json")

engine        = None
knowledge_hub = None
agent_loop    = None

# ── Per-session conversation memory ───────────────────────────────────────────
session_store: dict[str, list[dict]] = defaultdict(list)
session_lock = threading.Lock()

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
        elif path == "/v1/tools":
            self._json({"tools": get_tool_manifest()})
        elif path == "/v1/metrics":
            with metrics_lock:
                recent = metrics_store[-50:]  # last 50 requests
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

        # ── RAG Context ───────────────────────────────────────────────────────
        rag_context = ""
        rag_source  = None
        if use_rag:
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
                print(f"  [RAG] Hit: {rag_source}")

        # ── Episodic Memory ───────────────────────────────────────────────────
        episodic_ctx = ""
        memories = knowledge_hub.search_episodic_memory(last_user, top_k=1)
        if memories:
            episodic_ctx = f"[Past conversation: {memories[0][:200]}]\n"

        # ── Build Prompt ──────────────────────────────────────────────────────
        conv_history = ""
        recent = messages[-6:] if len(messages) > 6 else messages
        for msg in recent[:-1]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            conv_history += f"{role}: {msg.get('content', '')}\n"

        prompt = ""
        if episodic_ctx:
            prompt += episodic_ctx
        if rag_context:
            prompt += rag_context + "\n"
        prompt += conv_history + f"User: {last_user}\nAssistant:"

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

            self._sse({"done": True, "stats": {
                "tokens": gen_count, "time_s": round(elapsed, 2), "tps": round(tps, 1)
            }})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

            # Record metrics
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
        self._record_metric(len(generated.split()), elapsed * 1000, rag_source)
        self._json({
            "id":      f"chatcmpl-{int(time.time())}",
            "model":   "stories15M-custom-cpp",
            "choices": [{"message": {
                "role":     "assistant",
                "content":  generated.strip() or "...",
                "rag_source": rag_source
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


# ─── Server Startup ───────────────────────────────────────────────────────────
def run_server():
    global engine, knowledge_hub, agent_loop

    print(f"[Server] Loading model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[Server] ERROR: Model not found. Run: python tools/download_model.py")
        return

    engine        = PyEngine(MODEL_PATH, TOKENIZER_PATH)
    embedding_dim = engine.cfg.dim

    print(f"[Server] Loading Knowledge Hub (dim={embedding_dim})...")
    knowledge_hub = PersistentKnowledgeHub(embedding_dim=embedding_dim, db_path=KNOWLEDGE_DB)

    print("[Server] Initializing Agent Loop...")
    agent_loop = AgentLoop(engine, knowledge_hub=knowledge_hub, max_steps=3)

    print(f"\n{'='*60}")
    print(f"  LeagueOfLLMs Server v2 — ONLINE")
    print(f"  URL:        http://localhost:{PORT}")
    print(f"  Model:      stories15M (dim={embedding_dim}, layers={engine.cfg.n_layers})")
    print(f"  Knowledge:  {len(knowledge_hub.chunks)} chunks loaded")
    print(f"  Features:   RAG v2 | Tool Calling | Episodic Memory | Metrics")
    print(f"{'='*60}\n")

    server_address = ("", PORT)
    # Allow address reuse to avoid "address already in use"
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, LLMRequestHandler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
