"""OpenAI-compatible chat endpoint backed by headless Claude Code (claude -p, subscription, no API cost).

The tau2 harness sends chat completions (with tools for the agent, without for the user simulator). Each request
becomes one `claude -p` call without tools: the conversation and tool schemas go in as text, and the model answers
with JSON (a message or tool calls), which is turned back into an OpenAI response.

  python claude_proxy.py [port]   model field "<tag>|<sonnet|haiku|opus>"; log in calls.jsonl
"""
import json, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
HERE = Path(__file__).resolve().parent
lock = threading.Lock()
SYS_AGENT = ("You are the assistant in a customer-service conversation. You will get your instructions, the tools you "
             "can call and the conversation so far. Decide your next turn. Answer with JSON only, no prose around it: "
             '{"type": "message", "content": "<text to the customer>"} or '
             '{"type": "tool_calls", "calls": [{"name": "<tool>", "arguments": {...}}]}. '
             "Never both. Tool arguments must follow the schemas exactly.")
SYS_USER = ("You play the customer in a customer-service conversation, following the scenario in your instructions. "
            "Answer with the customer's next message only, plain text, nothing else.")


def render(msgs):
    out = []
    for m in msgs:
        r = m.get("role")
        if r == "system":
            continue
        if r == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                f = tc.get("function", {})
                out.append(f"ASSISTANT CALLS TOOL {f.get('name')} WITH {f.get('arguments')}")
            if m.get("content"):
                out.append(f"ASSISTANT: {m['content']}")
        elif r == "tool":
            out.append(f"TOOL RESULT: {m.get('content')}")
        else:
            out.append(f"{r.upper()}: {m.get('content')}")
    return "\n".join(out)


def call(model, system, prompt):
    cmd = ["claude", "-p", "--model", model, "--tools", "", "--setting-sources", "", "--no-session-persistence",
           "--output-format", "json", "--system-prompt", system]
    for attempt in range(3):
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600, cwd="/tmp")
        try:
            d = json.loads(r.stdout)
            if not d.get("is_error") and d.get("result"):
                return d["result"], d
        except Exception:
            pass
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"claude -p failed: {r.stdout[-200:]} {r.stderr[-200:]}")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        tag, model = body["model"].split("|", 1)
        msgs = body.get("messages", []); tools = body.get("tools") or []
        sysmsg = "\n\n".join(m["content"] for m in msgs if m.get("role") == "system" and m.get("content"))
        t0 = time.time()
        if tools:
            schemas = json.dumps([t["function"] for t in tools], indent=0)
            # static part (instructions + tool schemas) in the system prompt, so Claude's prompt cache reuses it across turns
            system = f"{SYS_AGENT}\n\nYOUR INSTRUCTIONS:\n{sysmsg}\n\nTOOLS (JSON schemas):\n{schemas}"
            prompt = f"CONVERSATION SO FAR:\n{render(msgs)}\n\nYour next turn, as JSON:"
            msg = None
            for _ in range(3):
                txt, meta = call(model, system, prompt)
                try:
                    j = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
                    if j.get("type") == "tool_calls" and j.get("calls"):
                        msg = {"role": "assistant", "content": None, "tool_calls": [
                            {"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                             "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}} for c in j["calls"]]}
                    elif j.get("content"):
                        msg = {"role": "assistant", "content": j["content"]}
                    if msg: break
                except Exception:
                    continue
            if msg is None:
                msg = {"role": "assistant", "content": txt.strip() or "Sorry, could you repeat that?"}
            finish = "tool_calls" if msg.get("tool_calls") else "stop"
        else:
            prompt = f"CONVERSATION SO FAR (you are USER):\n{render(msgs)}\n\nYour next message:"
            txt, meta = call(model, f"{SYS_USER}\n\nYOUR INSTRUCTIONS:\n{sysmsg}", prompt)
            msg = {"role": "assistant", "content": txt.strip()}; finish = "stop"
        with lock, open(HERE / "calls.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), "tag": tag, "model": model, "sec": round(time.time() - t0, 1),
                                "notional_usd": meta.get("total_cost_usd"), "tool": bool(tools), "cache_read": (meta.get("usage") or {}).get("cache_read_input_tokens"), "cache_write": (meta.get("usage") or {}).get("cache_creation_input_tokens")}) + "\n")
        self.reply({"id": "c" + uuid.uuid4().hex[:10], "object": "chat.completion", "model": model,
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8097
    print(f"claude proxy on 127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
