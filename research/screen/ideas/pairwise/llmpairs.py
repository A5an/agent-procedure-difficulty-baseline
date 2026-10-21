"""Cached pairwise judge. python llmpairs.py [test|run]"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import hashlib, json, os, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
REPO = Path(ZR.REPO)
sys.path.insert(0, str(REPO / "src"))
os.environ["LLM_BACKEND"] = "vertex"
os.environ["GOOGLE_CLOUD_PROJECT"] = subprocess.run(["gcloud","config","get-value","project"],capture_output=True,text=True).stdout.strip()
import llm
HERE = Path(__file__).parent
CACHE = HERE / "cache.jsonl"
POL = Path(ZR.DATA + "/tau2-domains/tasks")
POLICY = {"airline": "airline_policy.md", "retail": "retail_policy.md", "telecom": "telecom_main_policy.md"}
MAXCALLS = 5500
LOCK = threading.Lock(); calls = {"n": 0}
TEXT = {b: {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(REPO/"data"/b/"statements.jsonl")} for b in ["sopbench","tau2"]}
Q = ('Which case is an AI customer-service agent with tool access more likely to handle incorrectly '
     '(break a policy rule, take the wrong final action, or refuse when it should act / act when it should refuse)? '
     'Answer JSON {"harder": "A" or "B", "confidence": number from 0.5 to 1.0}')
def prompt(bench, dom, a, b):
    head = "You compare two customer-service cases from the same domain.\n\n"
    if bench == "tau2":
        head += "Domain policy that the agent must follow:\n<policy>\n" + (POL/POLICY[dom]).read_text() + "\n</policy>\n\n"
    return head + f"Case A:\n{TEXT[bench][a]}\n\nCase B:\n{TEXT[bench][b]}\n\n{Q}"
def key(p): return hashlib.sha256(p.encode()).hexdigest()
def parse(t):
    t = t.strip(); i, j = t.find("{"), t.rfind("}")
    d = json.loads(t[i:j+1])
    if d.get("harder") not in ("A","B"): raise ValueError
    c = float(d["confidence"]); 
    return {"harder": d["harder"], "confidence": min(1.0, max(0.5, c))}
def call(row):
    p = prompt(row.bench, row.domain, row.a, row.b)
    for att in range(3):
        with LOCK:
            if calls["n"] >= MAXCALLS: raise SystemExit("budget")
            calls["n"] += 1
        try:
            out, _ = llm.generate(p + ("" if att == 0 else "\nReturn only the JSON object."), max_tokens=1500)
            d = parse(out)
            return {"key": key(p), "pair_id": int(row.pair_id), **d, "ok": True}
        except SystemExit: raise
        except Exception as e: last = type(e).__name__
    return {"key": key(p), "pair_id": int(row.pair_id), "ok": False, "err": last}
def done():
    s = set()
    if CACHE.exists():
        for l in open(CACHE):
            r = json.loads(l)
            if r.get("ok"): s.add(r["key"])
    return s
def run(pairs, workers=8):
    d = done(); todo = [r for r in pairs.itertuples() if key(prompt(r.bench, r.domain, r.a, r.b)) not in d]
    print("to do", len(todo), flush=True); n = 0
    with ThreadPoolExecutor(workers) as ex, open(CACHE, "a") as f:
        for rec in ex.map(call, todo):
            f.write(json.dumps(rec)+"\n"); f.flush(); n += 1
            if n % 250 == 0: print(n, "calls", calls["n"], flush=True)
    print("finished, calls", calls["n"])
if __name__ == "__main__":
    P = pd.read_csv(HERE/"pairs.csv").fillna({"repeat_of": ""})
    if sys.argv[1] == "test":
        s = pd.concat([P[P.bench=="sopbench"].head(3), P[P.bench=="tau2"].head(3)])
        for r in s.itertuples(): print(call(r))
    else:
        order = P[P.repeat_of == ""]; run(order); run(P[P.repeat_of != ""])
