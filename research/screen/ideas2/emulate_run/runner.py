"""Sharded, resumable SOPBench runner. Each job = a chunk of one domain's cases, run by a run_simulation.py subprocess
(one model call at a time per process, so workers = parallel calls), then scored by run_evaluation.py (SOPBench's own evaluator)."""
import sys, os, json, subprocess, copy, time, glob, argparse, threading
from concurrent.futures import ThreadPoolExecutor
H = os.path.dirname(os.path.abspath(__file__)); SOP = H + "/sop"
PY = H + "/venv/bin/python"
PRICE = {"gemini-2.5-flash-lite": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50)}   # USD per 1M input, output (thinking included)
FN = "ast_{m}-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json"

def cost_of(usage_files, model):
    pi, po = PRICE[model]; tot = 0.0; n = 0
    for f in usage_files:
        if not os.path.exists(f): continue
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            tot += r["prompt"] * pi / 1e6 + (r["cand"] + r["thoughts"]) * po / 1e6; n += 1
    return tot, n

def make_jobs(run, tasks_dir, chunk, only=None):
    """tasks_dir has <domain>_tasks.json. Adds a stable case id and splits each domain into chunks."""
    jobs = []
    for dom in ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]:
        f = f"{tasks_dir}/{dom}_tasks.json"
        if not os.path.exists(f): continue
        d = json.load(open(f)); flat = []
        for proc, cs in d.items():
            for i, c in enumerate(cs):
                c = copy.deepcopy(c); c["_cid"] = c.get("_cid") or f"{dom}/{proc}/{i}"; flat.append((proc, c))
        if only: flat = [(p, c) for p, c in flat if only(dom, p, c)]
        for k in range(0, len(flat), chunk):
            sub = {}
            for p, c in flat[k:k + chunk]: sub.setdefault(p, []).append(c)
            jid = f"{dom}_{k // chunk:03d}"
            sd = f"{H}/runs/{run}/shards/{jid}"; os.makedirs(sd, exist_ok=True)
            json.dump(sub, open(f"{sd}/{dom}_tasks.json", "w"))
            jobs.append((jid, dom, sd))
    return jobs

def run_job(job, run, model, max_tokens, cap, state):
    jid, dom, sd = job
    out = f"{H}/runs/{run}/out/{jid}"; os.makedirs(out, exist_ok=True)
    fin = f"{out}/{dom}/" + FN.format(m=model)
    done_flag = f"{out}/DONE"
    if os.path.exists(done_flag): return
    with state["lock"]:
        if state["stop"]: return
        c, _ = cost_of(glob.glob(f"{H}/runs/*/usage_*.jsonl") if False else state["usage_files"](), model)
        if c > cap: state["stop"] = True; print("BUDGET CAP reached", c, flush=True); return
    env = dict(os.environ, USAGE_LOG=f"{H}/runs/{run}/usage_{jid}.jsonl", PYTHONHASHSEED="0")
    env["LLM_BACKEND"] = "vertex"
    common = ["--domain", dom, "--assistant_model", model, "--output_dir", out, "--tool_list", "full", "--tool_call_mode", "fc"]
    log = open(f"{out}/log.txt", "a")
    r = subprocess.run([PY, "run_simulation.py", "--data_dir", sd, "--assistant_temperature", "0", "--assistant_max_tokens", str(max_tokens)] + common,
                       cwd=SOP, env=env, stdout=log, stderr=subprocess.STDOUT)
    r2 = subprocess.run([PY, "run_evaluation.py"] + common, cwd=SOP, env=env, stdout=log, stderr=subprocess.STDOUT)
    open(done_flag, "w").write(f"{r.returncode} {r2.returncode}")
    print("done", jid, r.returncode, r2.returncode, flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run"); ap.add_argument("--tasks_dir"); ap.add_argument("--model"); ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=20); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--cap", type=float, default=10.0)
    ap.add_argument("--proc_filter", default=None, help="json file: list of 'dom/proc' to keep")
    a = ap.parse_args()
    keep = set(json.load(open(a.proc_filter))) if a.proc_filter else None
    only = (lambda d, p, c: f"{d}/{p}" in keep) if keep else None
    jobs = make_jobs(a.run, a.tasks_dir, a.chunk, only)
    os.makedirs(f"{H}/runs/{a.run}", exist_ok=True)
    state = {"lock": threading.Lock(), "stop": False, "usage_files": lambda: glob.glob(f"{H}/runs/{a.run}/usage_*.jsonl")}
    print(len(jobs), "jobs", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda j: run_job(j, a.run, a.model, a.max_tokens, a.cap, state), jobs))
    c, n = cost_of(state["usage_files"](), a.model); print("COST", round(c, 3), "calls", n)
