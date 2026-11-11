"""Local OpenAI-compatible proxy (copied from ../synth/proxy.py on 4 Nov for the replication runs; own ledger and cap file in this folder;
ledger-based cap, usd per call) from a harness to Gemini on Vertex AI.

The harness posts chat completions to http://127.0.0.1:PORT/v1/chat/completions with model "<tag>|<gemini model>".
The proxy adds a fresh gcloud access token, sends gemini-2.5-* calls round robin over regional endpoints (the global
endpoint answered 429 most of the time on 13 Oct) and gemini-3.8-flash to global, retries 429 and 5xx, and writes the
token usage of every call to the ledger. Calls tagged as the user simulator or the judge ("...user" or "...judge") get
a thinking budget of 0 (reasoning off); reasoning_effort "minimal" still spent about 120 thinking tokens, which emptied
the judge's 128-token answers in the smoke test.

  python vertex_proxy.py [port]        (default 8091; ledger: $TB_LEDGER or run/ledger.jsonl)
The Google Cloud project is read from `gcloud config get-value project` and never printed or logged."""
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests
import urllib3.util.connection as uc

uc.HAS_IPV6 = False  # hung IPv6 connections on 13 Oct

HERE = Path(__file__).resolve().parent
LEDGER = Path(os.environ.get("TIB_LEDGER", HERE / "ledger.jsonl"))
EU = ["europe-west4", "europe-west1", "europe-north1"]
REG = {"gemini-2.5-flash-lite": EU,
       "gemini-2.5-flash": EU + ["europe-west3", "europe-west8", "europe-west9", "europe-southwest1", "europe-central2",
                                 "northamerica-northeast1", "us-south1", "us-west4"],
       "gemini-2.5-pro": EU + ["europe-west8", "europe-southwest1", "europe-central2", "northamerica-northeast1",
                               "us-south1", "us-west4", "us-west1", "us-central1", "us-east1"],
       "gemini-3.8-flash": ["global"]}
_rr = {m: itertools.cycle(r) for m, r in REG.items()}
SIG = {}  # (tool_call id, name, arguments) -> extra_content with the Gemini 3 thought signature
_lock = threading.Lock()
_tok = {"v": None, "t": 0.0}
PRICE = {"gemini-2.5-flash-lite": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-pro": (1.25, 10.00),
         "gemini-3.8-flash": (0.75, 3.75)}
CAP = float(os.environ.get("TIB_CAP_USD", "45.0"))  # budget 45 USD for the replication (Tima, 4 Nov)
CAP_FILE = HERE / "cap_usd.txt"  # if present, its number replaces CAP at every call (raise the cap without a restart)
def _ledger_total():
    tot = 0.0
    if LEDGER.exists():
        for l in open(LEDGER):
            try: tot += json.loads(l).get("usd", 0.0)
            except Exception: pass
    return tot


_spent = {"usd": _ledger_total()}  # the cap counts everything already in the ledger, across restarts
# Agent turn limit: an agent request whose conversation already holds MAX_CALLS assistant tool-call turns gets a final
# text reply instead of a model call, which ends the agent's loop (added during the main run of 15 Oct after a
# gemini-2.5-pro episode repeated one tool call hundreds of times; no finished episode had more than 36 such turns).
MAX_CALLS = int(os.environ.get("TIB_MAX_AGENT_CALLS", "60"))
STOP = {"id": "proxy-turn-limit", "object": "chat.completion", "model": "",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "I am unable to complete this request. <DONE>"}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
PROJECT = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True).stdout.strip()


def token(force=False):
    with _lock:
        if force or _tok["v"] is None or time.time() - _tok["t"] > 2400:
            _tok["v"] = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True,
                                       text=True).stdout.strip()
            _tok["t"] = time.time()
        return _tok["v"]


def region(model):
    with _lock:
        return next(_rr[model])


_local = threading.local()


def session():
    # one keep-alive session per handler thread: fewer TLS handshakes than a new connection per call
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def url(loc):
    host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{PROJECT}/locations/{loc}/endpoints/openapi/chat/completions"


def log(rec):
    with _lock, open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        tag, model = body["model"].split("|", 1) if "|" in body["model"] else ("", body["model"])
        if model not in REG:
            return self.reply(400, json.dumps({"error": {"message": f"unknown model {model}"}}).encode())
        cap = CAP
        try:
            cap = float(CAP_FILE.read_text().strip())
        except (OSError, ValueError):
            pass
        if cap and _spent["usd"] >= cap:
            return self.reply(429, json.dumps({"error": {"message": f"proxy spending cap {cap} USD reached"}}).encode())
        if tag.endswith("agent") and sum(1 for m in body.get("messages", [])
                                         if m.get("role") == "assistant" and m.get("tool_calls")) >= MAX_CALLS:
            log({"t": round(time.time(), 1), "tag": tag, "model": model, "turn_limit": True, "in": 0, "out": 0,
                 "cached": 0, "reasoning": 0})
            return self.reply(200, json.dumps(STOP).encode())
        body["model"] = "google/" + model
        if model.startswith("gemini-3"):
            # Gemini 3 wants each earlier function call sent back with its thought signature; litellm drops the
            # extra_content field, so re-attach the signature cached from our own earlier response (dummy if unknown)
            for m in body.get("messages", []):
                for tc in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
                    if not tc.get("extra_content"):
                        f = tc.get("function") or {}
                        tc["extra_content"] = SIG.get((tc.get("id"), f.get("name"), f.get("arguments")),
                                                      {"google": {"thought_signature": "skip_thought_signature_validator"}})
        if tag.endswith(("user", "judge")):
            body.pop("reasoning_effort", None)
            body["extra_body"] = {"google": {"thinking_config": {"thinking_budget": 0}}}
        t0, last = time.time(), (503, b'{"error": {"message": "no attempt"}}')
        empty_retries = 0
        for attempt in range(14):
            loc = region(model)
            try:
                r = session().post(url(loc), json=body, headers={"Authorization": f"Bearer {token()}"},
                                   timeout=(20, 300))
            except Exception as e:
                detail = re.sub(r"projects/[^/\s'\"`]+", "projects/<P>", str(e).replace(PROJECT, "<P>"))[:160]
                last = (503, json.dumps({"error": {"message": f"proxy: {type(e).__name__}: {detail}"}}).encode())
                _local.__dict__.pop("s", None)
                time.sleep(min(60, 2 ** attempt))
                continue
            if r.status_code == 200:
                j = r.json()
                u = j.get("usage") or {}
                ch = (j.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                for tc in msg.get("tool_calls") or []:
                    if tc.get("extra_content"):
                        f = tc.get("function") or {}
                        with _lock:
                            SIG[(tc.get("id"), f.get("name"), f.get("arguments"))] = tc["extra_content"]
                empty = not msg.get("content") and not msg.get("tool_calls")
                if empty and os.environ.get("TB_DEBUG_EMPTY"):
                    d = HERE / "debug"
                    d.mkdir(exist_ok=True)
                    (d / f"{time.time():.3f}_{model}.json").write_text(json.dumps({"request": body, "response": j}))
                pi, po = PRICE[model]
                cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                usd = ((u.get("prompt_tokens", 0) - cached) * pi + cached * pi * 0.1 + (
                    u.get("completion_tokens", 0) + (u.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens", 0)) * po) / 1e6
                with _lock:
                    _spent["usd"] += usd
                log({"usd": round(usd, 6), "finish": ch.get("finish_reason"), "empty": empty, "t": round(time.time(), 1), "tag": tag, "model": model, "loc": loc, "attempts": attempt + 1,
                     "sec": round(time.time() - t0, 1), "in": u.get("prompt_tokens", 0),
                     "out": u.get("completion_tokens", 0),
                     "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                     "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)})
                # an empty agent reply (no text, no tool call) crashes the tau2 orchestrator; flash-lite gave one in the
                # 3 Nov smoke test. Ask again, at most 2 more times (each attempt is billed and logged above).
                if empty and tag.endswith("agent") and empty_retries < 4:
                    # at temperature 0 (and a fixed seed) the same request gives the same empty reply, so the retries
                    # sample at 0.5, then 1.0, with a changed seed
                    empty_retries += 1
                    body["temperature"] = min(1.0, 0.5 * empty_retries)
                    if "seed" in body:
                        body["seed"] = int(body["seed"]) + empty_retries
                    continue
                return self.reply(200, r.content)
            last = (r.status_code, re.sub(rb"projects/[^/\s\"`']+", b"projects/<P>", r.content.replace(PROJECT.encode(), b"<P>")))
            if r.status_code == 401:
                token(force=True)
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))
                continue
            break
        log({"t": round(time.time(), 1), "tag": tag, "model": model, "error": last[0],
             "msg": last[1][:300].decode(errors="replace")})
        self.reply(*last)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8091
    print(f"vertex proxy on 127.0.0.1:{port}, ledger {LEDGER}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
