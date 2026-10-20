"""Cached FMEA runner. python llmrun.py test | main | retest | banking"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, os, sys, threading, random, subprocess, hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(ZR.REPO)
POL = Path(ZR.DATA + "/tau2-domains/tasks")
sys.path.insert(0, str(REPO / "src"))
os.environ["LLM_BACKEND"] = "vertex"
os.environ["GOOGLE_CLOUD_PROJECT"] = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True).stdout.strip()
import llm  # noqa
import prompts as P
HERE = Path(__file__).parent
CACHE = HERE / "cache" / "fmea.jsonl"
LOCK = threading.Lock(); calls = {"n": 0}; MAXCALLS = 1500
POLICY = {"airline": "airline_policy.md", "retail": "retail_policy.md", "telecom": "telecom_main_policy.md"}

def statements(b): return [json.loads(l) for l in open(REPO / "data" / b / "statements.jsonl")]
def policy_for(b, cid):
    if b != "tau2": return None
    return (POL / POLICY[cid.split("/")[0]]).read_text()
def shuffled(seed): o = list(P.CATS); random.Random(seed).shuffle(o); return o
def parse(t):
    t = t.strip(); a, b = t.find("{"), t.rfind("}"); return json.loads(t[a:b + 1])
def valid(d):
    if not isinstance(d, dict) or not isinstance(d.get("modes"), list) or len(d["modes"]) > 8: return False
    for m in d["modes"]:
        if m.get("category") not in P.CATS: return False
        for k in ("occurrence", "severity", "detection"):
            if not isinstance(m.get(k), (int, float)) or not 1 <= m[k] <= 10: return False
    return True

def call(variant, b, cid, text, order):
    prompt = P.prompt(text, order, policy_for(b, cid))
    key = f"{variant}|{cid}"; last = ""
    for att in range(3):
        p = prompt if att == 0 else prompt + "\n\nYour previous answer was not valid. Return exactly one valid JSON object as specified, nothing else."
        with LOCK:
            if calls["n"] >= MAXCALLS: raise SystemExit("budget")
            calls["n"] += 1
        try:
            out, _ = llm.generate(p, max_tokens=4000); last = out
            d = parse(out)
            if valid(d): return {"key": key, "parsed": d, "ok": True}
        except SystemExit: raise
        except Exception as e: last = f"ERR {type(e).__name__}"
    return {"key": key, "parsed": None, "raw": last[:300], "ok": False}

def run(variant, items, order):
    done = set()
    if CACHE.exists():
        done = {json.loads(l)["key"] for l in open(CACHE) if json.loads(l).get("ok")}
    todo = [(b, c, t) for b, c, t in items if f"{variant}|{c}" not in done]
    print(variant, "todo", len(todo), flush=True); n = 0
    with ThreadPoolExecutor(8) as ex, open(CACHE, "a") as f:
        for r in ex.map(lambda x: call(variant, *x, order), todo):
            f.write(json.dumps(r) + "\n"); f.flush(); n += 1
            if n % 100 == 0: print(n, "calls", calls["n"], flush=True)

def items(benches): return [(b, r["case_id"], r["text"]) for b in benches for r in statements(b)]
def sample100():
    a = items(["sopbench", "tau2"]); random.Random(0).shuffle(a); return a[:100]

if __name__ == "__main__":
    w = sys.argv[1]
    if w == "test":
        for b in ("sopbench", "tau2"):
            r = statements(b)[3]; x = call("test", b, r["case_id"], r["text"], P.CATS); print(b, json.dumps(x)[:1500])
    elif w == "main": run("A", items(["sopbench", "tau2"]), P.CATS)
    elif w == "retest": run("B", sample100(), shuffled(7))
    elif w == "banking": run("A", items(["tauk_banking"]), P.CATS)
