"""Forecaster panel: D prompt on several models. usage: panel.py gemini|haiku|opus"""
import sys, json, os, hashlib, subprocess, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "ablate")); sys.path.insert(0, str(HERE.parent / "common"))
import ablate as A
import vx
JOBS = {k: p for (kind, k), p in A.jobs.items() if kind == "D"}   # key -> prompt (tau2, banking, sopbench, tac, mcpmark, drafter)
def bench(c): return c.split("::")[0] if "::" in c else ("sopbench" if c.count("/") == 2 else ("tauk_banking" if c.startswith("banking") else "tau2"))
lock = threading.Lock()
def claude_call(model, cache, path, prompt):
    k = hashlib.sha256(prompt.encode()).hexdigest()
    if k in cache: return cache[k]
    cmd = ["claude", "-p", "--model", model, "--tools", "", "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
    text = None
    for a in range(5):
        try: res = json.loads(subprocess.run(cmd, input=prompt, cwd=HERE / "work", capture_output=True, text=True, timeout=600).stdout)
        except Exception as e: res = {"is_error": True, "result": str(e)}
        if not res.get("is_error") and res.get("result"): text = res["result"]; break
        time.sleep(20 * (a + 1))
    if text:
        with lock:
            cache[k] = text
            with open(path, "a") as f: f.write(json.dumps({"k": k, "r": text, "cost": res.get("total_cost_usd")}) + "\n")
    return text
if __name__ == "__main__":
    w = sys.argv[1]
    if w == "gemini":
        C = vx.Client(HERE)
        keys = [k for k in JOBS if bench(k) != "sopbench"]
        jobs = [(JOBS[k], m) for k in keys for m in ["gemini-2.5-pro", "gemini-3.5-flash", "gemini-3.8-flash"]]
        uniq = list(dict.fromkeys(jobs))
        with ThreadPoolExecutor(16) as ex: list(ex.map(lambda j: C.call(j[0], model=j[1], tag="D", max_tokens=4000), uniq))
        out = {m: {k: C.call(JOBS[k], model=m, tag="D", max_tokens=4000) for k in keys} for m in ["gemini-2.5-pro", "gemini-3.5-flash", "gemini-3.8-flash"]}
        json.dump(out, open(HERE / "gemini_raw.json", "w")); print("done gemini", C.ncalls, "fails", C.nfail)
    else:
        path = HERE / f"{w}_cache.jsonl"; cache = {}
        if path.exists():
            for l in path.read_text().splitlines(): d = json.loads(l); cache[d["k"]] = d["r"]
        keys = [k for k in JOBS if bench(k) in (("tau2", "tauk_banking", "tac", "mcpmark") if w == "opus" else ("tau2", "tauk_banking", "tac", "mcpmark", "drafter"))]
        ps = list(dict.fromkeys(JOBS[k] for k in keys))
        with ThreadPoolExecutor(int(os.environ.get("NW", "8"))) as ex: list(ex.map(lambda p: claude_call(w, cache, path, p), ps))
        json.dump({k: cache.get(hashlib.sha256(JOBS[k].encode()).hexdigest()) for k in keys}, open(HERE / f"{w}_raw.json", "w")); print("done", w)
