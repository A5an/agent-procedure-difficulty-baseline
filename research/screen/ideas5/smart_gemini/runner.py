import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os, sys, json, csv, glob, time, threading, argparse
H2 = ZR.SCREEN + "/ideas2/harness"
MINE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H2)
import importlib.util
spec = importlib.util.spec_from_file_location("hrun", H2 + "/run.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
from swarm import vclient
from swarm.vclient import ctx, BudgetExceeded
from concurrent.futures import ThreadPoolExecutor
vclient.LEDGER = MINE + "/ledger.jsonl"
MODEL = "gemini-3.8-flash"
EP = MINE + "/episodes"; os.makedirs(EP, exist_ok=True)

def synth_tasks():
    out = {}
    for d in ["bank","dmv","healthcare","hotel","library","online_market","university"]:
        td = json.load(open(f"{H2}/../emulate_run/synth_all/{d}_tasks.json"))
        for goal, arr in td.items():
            for t in arr:
                t = dict(t); t["user_goal"] = goal; t["domain"] = d; out[t["_cid"]] = t
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cap", type=float, default=55.0); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    vclient.CAPS["total"] = a.cap; vclient.load_ledger()
    if a.set == "real":
        tasks = R.load_tasks(); ids = list(tasks)
        pre = {}
        for l in open(f"{H2}/episodes/{MODEL}__H0.jsonl"):
            r = json.loads(l)
            if not r.get("error"): pre[r["case_id"]] = r
        path = f"{EP}/real.jsonl"
        if not os.path.exists(path):
            with open(path, "w") as f:
                for r in pre.values(): r = dict(r); r["reused"] = True; f.write(json.dumps(r, default=str) + "\n")
    else:
        tasks = synth_tasks(); ids = list(tasks); path = f"{EP}/synth.jsonl"
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if not r.get("error"): done.add(r["case_id"])
    todo = [c for c in ids if c not in done]
    if a.limit: todo = todo[:a.limit]
    print(a.set, len(ids), "cases", len(todo), "todo", "spent", round(vclient.spent(), 2), flush=True)
    lock = threading.Lock(); stop = {"v": False}; units = progs = None
    def work(cid):
        if stop["v"]: return
        t0 = time.time(); err = None; res = None
        for attempt in range(3):
            try:
                res, _ = R.run_episode(MODEL, "gemini", "global", "H0", cid, tasks[cid], {}, {}, 4096, 0.0); err = None; break
            except BudgetExceeded as e: stop["v"] = True; err = "BUDGET " + str(e); break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:300]}"; time.sleep(5 * (attempt + 1))
        rec = {"case_id": cid, "model": MODEL, "arm": "H0", "error": err, "secs": round(time.time() - t0, 1), "usage": getattr(ctx, "usage", []), "max_tokens": 4096}
        if res and not err:
            rec["interaction"] = res["interaction"]; rec["database"] = res["database"]; rec["usd"] = sum(u["usd"] for u in rec["usage"])
        with lock:
            with open(path, "a") as f: f.write(json.dumps(rec, default=str) + "\n")
            print(cid, "ERR " + err if err else "ok", rec["secs"], round(vclient.spent(), 2), flush=True)
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(work, todo))
    print("finished", a.set, "spent", round(vclient.spent(), 2), flush=True)
main()
