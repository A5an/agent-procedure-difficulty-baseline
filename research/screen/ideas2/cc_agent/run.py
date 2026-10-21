import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os, sys, json, csv, subprocess, time, threading, argparse, shutil
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import setup_case as S
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)  # native | guard | test_native ...
ap.add_argument("--limit", type=int); ap.add_argument("--only"); ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
guard = "guard" in a.arm
rows = [r["case_id"] for r in csv.DictReader(open(ZR.SCREEN + "/ideas2/harness/sample.csv"))]
if a.only: rows = a.only.split(",")
if a.limit: rows = rows[:a.limit]
idx = {c: i for i, c in enumerate(r["case_id"] for r in csv.DictReader(open(ZR.SCREEN + "/ideas2/harness/sample.csv")))}
OUT = f"{HERE}/out/{a.arm}"; os.makedirs(OUT, exist_ok=True)
stop = {"v": False}; lock = threading.Lock(); consec = {"n": 0}
def work(cid):
    if stop["v"]: return
    k = f"s{idx[cid]:03d}"; jp = f"{OUT}/{k}.json"
    if os.path.exists(jp):
        try:
            if not json.load(open(jp)).get("is_error"): return
        except Exception: pass
    sd = f"{HERE}/sessions/{a.arm}/{k}"; wd = f"{HERE}/work/{a.arm}/{k}"
    shutil.rmtree(sd, ignore_errors=True); shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    S.make_session(cid, sd, guard)
    prompt = S.build_prompt(cid, sd)
    cmd = ["claude", "-p", prompt, "--model", "sonnet", "--permission-mode", "dontAsk", "--allowedTools", f"Bash(python3 {HERE}/sop_tool.py:*)",
           "--disallowedTools", "Read Edit Write Glob Grep WebFetch WebSearch Agent NotebookEdit Skill", "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
    t0 = time.time(); res = None
    for attempt in range(4):
        p = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, timeout=900)
        try: res = json.loads(p.stdout)
        except Exception: res = {"is_error": True, "result": "unparsable: " + p.stdout[:300] + p.stderr[:300]}
        txt = json.dumps(res)[:1500].lower()
        if res.get("is_error") and ("limit" in txt or "rate" in txt or "overloaded" in txt or "429" in txt):
            time.sleep(60 * (attempt + 1)); continue
        break
    res["case_id"] = cid; res["secs"] = round(time.time() - t0, 1)
    json.dump(res, open(jp, "w"), indent=1)
    with lock:
        if res.get("is_error"):
            consec["n"] += 1
            if consec["n"] >= 4: stop["v"] = True
        else: consec["n"] = 0
        print(cid, "ERR" if res.get("is_error") else "ok", res["secs"], res.get("num_turns"), res.get("total_cost_usd"), flush=True)
with ThreadPoolExecutor(a.workers) as ex: list(ex.map(work, rows))
print("done; stopped_early =", stop["v"])
