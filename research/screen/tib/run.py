"""Run agent episodes on TauIndianBankBench tasks or on our probes through the local Vertex proxy (PREREG.md).

  python run.py <agent: g38|flash> <set: pilot|real|probes|seeded> [concurrency]

One line per finished episode in runs/<set>_<agent>.jsonl (resumable: finished task ids are skipped), full messages in
runs/msgs/<set>_<agent>/<task_id>.json. Spending is capped by the proxy, not here.
"""
import json, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import HERE, AGENTS, USER_MODEL, PROXY, load_tasks

from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.runner import build_text_orchestrator, run_simulation

agent, which = sys.argv[1], sys.argv[2]
conc = int(sys.argv[3]) if len(sys.argv) > 3 else 6
sel = json.load(open(HERE / "selection.json"))
if which == "probes":
    tasks = json.load(open(HERE / "probes.json"))
elif which == "seeded":                      # exp 3, seeded/PREREG.md
    tasks = json.load(open(HERE / "seeded" / "probes.json"))
else:
    ids = sel["pilot_tasks"] if which == "pilot" else [i for f in sel["test_families"] for i in sel["real_tasks"][f]]
    allt = {t["id"]: t for t in load_tasks()}
    tasks = [allt[i] for i in ids]
out = HERE / "runs" / f"{which}_{agent}.jsonl"
msgdir = HERE / "runs" / "msgs" / f"{which}_{agent}"
msgdir.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    for l in open(out):
        r = json.loads(l)
        if r.get("reward") is not None: done.add(r["task_id"])
todo = [t for t in tasks if t["id"] not in done]
print(f"{which} {agent}: {len(tasks)} tasks, {len(done)} done, {len(todo)} to run", flush=True)

cfg = TextRunConfig(domain="indian_banking", agent="llm_agent",
                    llm_agent=f"openai/agent|{AGENTS[agent]}",
                    llm_args_agent={"api_base": PROXY, "api_key": "proxy", "temperature": 0.0, "max_tokens": 8192},
                    user="user_simulator", llm_user=f"openai/user|{USER_MODEL}",
                    llm_args_user={"api_base": PROXY, "api_key": "proxy", "temperature": 0.0, "max_tokens": 8192},
                    max_steps=200, max_errors=10, seed=300)


def one(t):
    t0 = time.time()
    try:
        task = Task.model_validate(t)
        res = run_simulation(build_text_orchestrator(cfg, task, seed=300))
        ri = res.reward_info
        rec = {"task_id": t["id"], "agent": agent, "set": which, "reward": float(ri.reward) if ri else None,
               "termination": str(res.termination_reason), "n_msgs": len(res.messages), "sec": round(time.time() - t0, 1),
               "db_ok": bool(ri.db_check.db_match) if ri and ri.db_check else None,
               "actions_ok": [bool(a.action_match) for a in (ri.action_checks or [])] if ri else None}
        (msgdir / f"{t['id']}.json").write_text(json.dumps([m.model_dump(mode="json") for m in res.messages]))
    except Exception as e:
        rec = {"task_id": t["id"], "agent": agent, "set": which, "reward": None, "error": f"{type(e).__name__}: {str(e)[:300]}",
               "sec": round(time.time() - t0, 1)}
        traceback.print_exc()
    return rec


with ThreadPoolExecutor(conc) as ex, open(out, "a") as f:
    futs = [ex.submit(one, t) for t in todo]
    for i, fu in enumerate(as_completed(futs)):
        rec = fu.result()
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(i + 1, rec["task_id"], rec.get("reward"), rec.get("termination", rec.get("error")), rec["sec"], flush=True)
