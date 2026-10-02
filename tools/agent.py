"""
Tool Calling & Agent Loop for LeagueOfLLMs

Implementasi ReAct (Reasoning + Acting) agent:
1. AI menghasilkan teks → deteksi pemanggilan tool
2. Eksekusi tool → inject hasil ke konteks
3. Lanjutkan generasi

Built-in tools:
  - calculator        : evaluasi ekspresi matematik
  - search_knowledge  : cari di vector knowledge base
  - search_web        : scrape dan tambahkan URL ke knowledge
  - get_time          : waktu & tanggal sekarang
  - count_words       : hitung kata dalam teks
  - define            : cari definisi kata (via wikipedia summary)
"""

import re
import json
import datetime
import urllib.request


# ─── Tool Decorator & Registry ────────────────────────────────────────────────

TOOL_REGISTRY: dict = {}

def tool(name: str, description: str, params: dict):
    """Decorator to register a tool function."""
    def decorator(fn):
        TOOL_REGISTRY[name] = {
            "fn":          fn,
            "description": description,
            "params":      params,
        }
        return fn
    return decorator


# ─── Built-in Tools ───────────────────────────────────────────────────────────

@tool(
    name="calculator",
    description="Evaluasi ekspresi matematika sederhana. Contoh: 2+2, sqrt(16), 3.14*5**2",
    params={"expression": "string — ekspresi matematika"}
)
def calculator(expression: str) -> str:
    # Only allow safe math operations
    safe_expr = re.sub(r'[^0-9+\-*/.()% \t\nsqrtabcdefghijklmnopqrstuvwxyz]', '', expression)
    allowed_names = {"sqrt": __import__("math").sqrt, "abs": abs,
                     "round": round, "pi": 3.14159265, "e": 2.71828182}
    try:
        result = eval(safe_expr, {"__builtins__": {}}, allowed_names)
        return str(round(float(result), 6))
    except Exception as ex:
        return f"Error: {ex}"


@tool(
    name="get_time",
    description="Dapatkan waktu dan tanggal saat ini.",
    params={}
)
def get_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%A, %d %B %Y — %H:%M:%S WIB")


@tool(
    name="count_words",
    description="Hitung jumlah kata dalam sebuah teks.",
    params={"text": "string — teks yang akan dihitung"}
)
def count_words(text: str) -> str:
    count = len(text.split())
    chars = len(text)
    return f"{count} words, {chars} characters"


@tool(
    name="define",
    description="Cari definisi/penjelasan singkat tentang suatu topik dari Wikipedia.",
    params={"topic": "string — kata atau topik yang dicari"}
)
def define(topic: str) -> str:
    try:
        encoded = urllib.request.quote(topic.replace(" ", "_"))
        url     = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req     = urllib.request.Request(url, headers={"User-Agent": "LeagueOfLLMs/1.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data.get("extract", "No definition found.")[:400]
    except Exception as e:
        return f"Error fetching definition: {e}"


@tool(
    name="list_tools",
    description="Tampilkan semua tools yang tersedia.",
    params={}
)
def list_tools() -> str:
    lines = []
    for name, info in TOOL_REGISTRY.items():
        params_str = ", ".join(f"{k}: {v}" for k, v in info["params"].items())
        lines.append(f"• {name}({params_str}) — {info['description']}")
    return "\n".join(lines)


# ─── Tool Call Parser ─────────────────────────────────────────────────────────

# Pattern: TOOL[tool_name](param1="value1", param2="value2")
TOOL_PATTERN = re.compile(
    r'TOOL\[(\w+)\]\(([^)]*)\)',
    re.IGNORECASE
)

def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from generated text."""
    calls = []
    for match in TOOL_PATTERN.finditer(text):
        tool_name  = match.group(1).lower()
        params_str = match.group(2).strip()

        # Parse key="value" pairs
        params = {}
        for pm in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', params_str):
            params[pm.group(1)] = pm.group(2)
        # Also parse key=value without quotes
        for pm in re.finditer(r'(\w+)\s*=\s*([^\s,)"]+)', params_str):
            if pm.group(1) not in params:
                params[pm.group(1)] = pm.group(2)

        calls.append({
            "raw":       match.group(0),
            "tool":      tool_name,
            "params":    params,
            "span_start": match.start(),
            "span_end":   match.end()
        })
    return calls


def execute_tool(tool_name: str, params: dict,
                 knowledge_hub=None) -> str:
    """Execute a tool by name with given params."""

    # Dynamic tools that need knowledge_hub
    if tool_name == "search_knowledge":
        if knowledge_hub is None:
            return "Knowledge base not available."
        query = params.get("query", "")
        results = knowledge_hub.retrieve(query, top_k=2, use_multihop=False)
        if not results:
            return "No relevant knowledge found."
        parts = []
        for r in results:
            parts.append(f"[{r.get('source', 'unknown')}]: {r['_context'][:300]}")
        return "\n\n".join(parts)

    if tool_name not in TOOL_REGISTRY:
        return f"Unknown tool: {tool_name}"

    fn = TOOL_REGISTRY[tool_name]["fn"]
    try:
        info = TOOL_REGISTRY[tool_name]
        # Map params to function args
        if not info["params"]:
            return fn()
        # Pass all parsed params as kwargs
        return fn(**params)
    except TypeError as e:
        return f"Tool call error ({tool_name}): {e}"
    except Exception as e:
        return f"Tool execution error: {e}"


# ─── Agent Loop ───────────────────────────────────────────────────────────────

class AgentLoop:
    """
    ReAct-style agent loop:
    1. Format prompt with tool documentation
    2. Generate text
    3. Parse and execute any tool calls
    4. Inject tool results and continue
    5. Repeat up to max_steps

    Tool call syntax (taught in system prompt):
      TOOL[calculator](expression="2+2")
      TOOL[search_knowledge](query="quantum computing")
      TOOL[get_time]()
    """

    SYSTEM_PROMPT = """\
You are LeagueOfLLMs, a helpful AI assistant with access to tools.
To use a tool, write TOOL[tool_name](param="value") in your response.
Available tools:
{tools_doc}
After a tool call, the result will be shown as [RESULT: ...].
Use results to answer the user. Be concise and factual.
"""

    def __init__(self, engine, knowledge_hub=None, max_steps: int = 4):
        self.engine        = engine
        self.knowledge_hub = knowledge_hub
        self.max_steps     = max_steps

    def _build_tools_doc(self) -> str:
        lines = []
        for name, info in TOOL_REGISTRY.items():
            params = ", ".join(f'{k}="{v}"' for k, v in info["params"].items())
            lines.append(f"  TOOL[{name}]({params}) → {info['description']}")
        # Add dynamic tool
        lines.append('  TOOL[search_knowledge](query="...") → Cari di knowledge base')
        return "\n".join(lines)

    def run(self, user_message: str, history: list[dict] = None,
            temperature: float = 0.7, max_tokens: int = 80) -> dict:
        """
        Run the agent for a user message.
        Returns: {"response": str, "tool_calls": list, "steps": int}
        """
        if history is None:
            history = []

        # Build initial prompt
        tools_doc = self._build_tools_doc()
        system    = self.SYSTEM_PROMPT.format(tools_doc=tools_doc)

        # Format conversation
        conv = ""
        for msg in history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            conv += f"{role}: {msg['content']}\n"
        conv += f"User: {user_message}\nAssistant:"

        prompt = system + "\n\n" + conv

        all_tool_calls = []
        full_response  = ""
        steps = 0

        while steps < self.max_steps:
            steps += 1

            # Generate with the engine
            tokens  = self.engine.tokenizer.encode(prompt)
            pos     = 0
            token   = tokens[0]
            prompt_len = len(tokens)
            generated = ""

            import numpy as np
            from py_reference import softmax

            for _ in range(max_tokens):
                logits = self.engine.forward(token, pos)
                if pos + 1 < prompt_len:
                    next_tok = tokens[pos + 1]
                else:
                    probs    = softmax(logits / temperature)
                    next_tok = int(np.random.choice(len(probs), p=probs))
                    if next_tok == 2:
                        break
                    word      = self.engine.tokenizer.decode(next_tok)
                    generated += word

                    # Early stop if we detect a tool call closing paren
                    if ")" in generated and "TOOL[" in generated:
                        break

                token = next_tok
                pos += 1

            full_response += generated

            # Check for tool calls in generated text
            calls = parse_tool_calls(generated)
            if not calls:
                # No tools → we're done
                break

            # Execute each tool and inject results
            injection = ""
            for call in calls:
                result = execute_tool(
                    call["tool"], call["params"],
                    knowledge_hub=self.knowledge_hub
                )
                injection += f"\n[RESULT: {result}]\n"
                all_tool_calls.append({
                    "tool":   call["tool"],
                    "params": call["params"],
                    "result": result
                })

            # Continue generation with result injected
            prompt = prompt + full_response + injection + "Assistant:"

        return {
            "response":   full_response.strip(),
            "tool_calls": all_tool_calls,
            "steps":      steps
        }


# ─── Tool Manifest (for API exposure) ─────────────────────────────────────────

def get_tool_manifest() -> list[dict]:
    manifest = []
    for name, info in TOOL_REGISTRY.items():
        manifest.append({
            "name":        name,
            "description": info["description"],
            "parameters":  info["params"]
        })
    # Add dynamic tool
    manifest.append({
        "name":        "search_knowledge",
        "description": "Cari informasi dalam knowledge base yang telah dipelajari",
        "parameters":  {"query": "string — pertanyaan atau topik yang dicari"}
    })
    return manifest
