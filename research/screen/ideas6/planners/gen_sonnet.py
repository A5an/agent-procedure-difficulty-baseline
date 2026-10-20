"""Sonnet 5.5 plans via headless claude -p (no tools), same prompts as Gemini plans. usage: gen_sonnet.py test|tau2|sop"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, hashlib, threading, time, subprocess, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
I3 = Path(ZR.SCREEN + "/ideas3")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import importlib.util
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m; sp.loader.exec_module(m); return m
# prompts: same builders as the Gemini runs (import does not call the LLM)
sys.path.insert(0, ZR.REPO + "/src")
G2 = load(I3 / "plan_tools/gen2.py", "gen2_mod")
G1 = load(I3 / "plan_tau/gen.py", "gen1_mod")
G2.TOOLS = json.load(open(HERE / "tool_text.json"))
CACHE = HERE / "sonnet_cache.jsonl"; lock = threading.Lock(); cache = {}
if CACHE.exists():
    for l in CACHE.read_text().splitlines():
        d = json.loads(l); cache[d["k"]] = d["r"]
WD = HERE / "work"; WD.mkdir(exist_ok=True)
stop = {"v": False}; consec = {"n": 0}; stats = {"calls": 0, "cost": 0.0}
def call(prompt):
    k = hashlib.sha256(prompt.encode()).hexdigest()
    if k in cache: return cache[k]
    if stop["v"]: return None
    cmd = ["claude", "-p", "--model", "sonnet", "--tools", "", "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
    text = None
    for attempt in range(6):
        try:
            p = subprocess.run(cmd, input=prompt, cwd=WD, capture_output=True, text=True, timeout=600)
            res = json.loads(p.stdout)
        except Exception as e:
            res = {"is_error": True, "result": f"exc {e}"}
        if not res.get("is_error") and res.get("result"):
            text = res["result"]; stats["cost"] += res.get("total_cost_usd") or 0; break
        txt = json.dumps(res)[:800].lower()
        time.sleep(min(300, 30 * (attempt + 1)) if ("limit" in txt or "rate" in txt or "overload" in txt or "429" in txt) else 10)
    with lock:
        stats["calls"] += 1
        if text is not None:
            cache[k] = text; consec["n"] = 0
            with open(CACHE, "a") as f: f.write(json.dumps({"k": k, "r": text}) + "\n")
        else:
            consec["n"] += 1
            if consec["n"] >= 6: stop["v"] = True
        if stats["calls"] % 20 == 0: print("calls", stats["calls"], "cost", round(stats["cost"], 2), flush=True)
    return text
def run(prompts, w=4):
    with ThreadPoolExecutor(w) as ex: return list(ex.map(call, prompts))
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("test", "tau2"):
        items = G2.build_all(); uniq = {}
        for c, p in items: uniq.setdefault(p, []).append(c)
        prompts = list(uniq)
        if mode == "test": prompts = [prompts[0]]
        out = run(prompts)
        if mode == "test": print(out[0][:2000]); sys.exit()
        res = {c: o for p, o in zip(prompts, out) for c in uniq[p]}
        json.dump(res, open(HERE / "sonnet_tau2.json", "w")); print("tau2 done", len(prompts), "missing", sum(v is None for v in res.values()))
    else:
        items = [i for i in G1.build_all() if i[0] == "sopbench"]
        uniq = {}
        for _, c, p in items: uniq.setdefault(p, []).append(c)
        prompts = list(uniq); print("sop cases", len(items), "distinct", len(prompts), flush=True)
        out = run(prompts)
        res = {c: o for p, o in zip(prompts, out) for c in uniq[p]}
        json.dump(res, open(HERE / "sonnet_sop.json", "w")); print("sop done missing", sum(v is None for v in res.values()))
