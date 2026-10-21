import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os, sys, json, csv, subprocess, time, threading, argparse, shutil, glob
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import setup_case as S
ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=["real", "synth"])
ap.add_argument("--limit", type=int); ap.add_argument("--only"); ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
TASKS = ZR.REPO + "/data/sopbench/tasks.csv"
def case_list(kind):
    if kind == "real":
        return [r["case_id"] for r in csv.DictReader(open(TASKS))]
    out = []
    for dom in ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]:
        d = json.load(open(f"{S.T.SYNTH}/{dom}_tasks.json"))
        for proc, cs in d.items():
            out += [c["_cid"] for c in cs]
    return out
ids = case_list(a.set); pre = "r" if a.set == "real" else "y"
key = {c: f"{pre}{i:04d}" for i, c in enumerate(ids)}
json.dump(key, open(f"{HERE}/keys_{a.set}.json", "w"))
rows = ids
if a.only: rows = a.only.split(",")
if a.limit: rows = rows[:a.limit]
OUT = f"{HERE}/out/{a.set}"; os.makedirs(OUT, exist_ok=True)
stop = {"v": False, "last_ok": time.time()}; lock = threading.Lock(); consec = {"n": 0}
LIMIT_WORDS = ("usage limit", "rate limit", "rate_limit", "overloaded", "429", "limit reached", "too many requests", "resets")
def work(cid):
    if stop["v"]: return
    k = key[cid]; jp = f"{OUT}/{k}.json"
    if os.path.exists(jp):
        try:
            if not json.load(open(jp)).get("is_error"): return
        except Exception: pass
    sd = f"{HERE}/sessions/{a.set}/{k}"; wd = f"{HERE}/work/{a.set}/{k}"
    t0 = time.time(); res = None
    attempt = 0
    while True:
        shutil.rmtree(sd, ignore_errors=True); shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
        S.make_session(cid, sd, a.set)
        prompt = S.build_prompt(cid, sd, a.set)
        cmd = ["claude", "-p", prompt, "--model", "sonnet", "--permission-mode", "dontAsk", "--allowedTools", f"Bash(python3 {HERE}/sop_tool.py:*)",
               "--disallowedTools", "Read Edit Write Glob Grep WebFetch WebSearch Agent NotebookEdit Skill", "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
        try:
            p = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=900)
            res = json.loads(p.stdout)
        except subprocess.TimeoutExpired:
            res = {"is_error": True, "result": "timeout"}
        except Exception:
            res = {"is_error": True, "result": "unparsable: " + p.stdout[:300] + p.stderr[:300]}
        txt = (str(res.get("result"))[:600] + str(res.get("api_error_status")) + (p.stderr[:300] if 'p' in dir() else "")).lower()
        if res.get("is_error") and any(w in txt for w in LIMIT_WORDS):
            attempt += 1
            with lock:
                if time.time() - stop["last_ok"] > 900: stop["v"] = True
            if stop["v"]: res["limit_stop"] = True; break
            print("limit-like error, backoff", cid, txt[:120], flush=True)
            time.sleep(min(60 * attempt, 300)); continue
        break
    res["case_id"] = cid; res["secs"] = round(time.time() - t0, 1)
    if res.get("limit_stop"): print("STOP: limit errors for 15 minutes", flush=True); return
    json.dump(res, open(jp, "w"), indent=1)
    with lock:
        if res.get("is_error"):
            consec["n"] += 1
            if consec["n"] >= 6: stop["v"] = True
        else: consec["n"] = 0; stop["last_ok"] = time.time()
        print(cid, "ERR" if res.get("is_error") else "ok", res["secs"], res.get("num_turns"), res.get("total_cost_usd"), flush=True)
with ThreadPoolExecutor(a.workers) as ex: list(ex.map(work, rows))
print("done; stopped_early =", stop["v"])
