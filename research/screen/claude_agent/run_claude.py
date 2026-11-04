"""TauIndianBank episodes with a Claude agent and a Claude user simulator through claude_proxy.py (subscription).

  python run_claude.py <agent model: sonnet|haiku> <set: real|seeded> <scope: old|new|all> [concurrency] [--only id,id]
Same tasks, probes, harness and grading as P0-2 / Exp 3 / the replication; only the models differ.
Output runs/<set>_<agent>.jsonl and runs/msgs/<set>_<agent>/.
"""
import json, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
HERE = Path(__file__).resolve().parent
S = HERE.parent
sys.path.insert(0, str(S / "tib"))
from common import load_tasks
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.runner import build_text_orchestrator, run_simulation

args = sys.argv[1:]
only = None
if "--only" in args:
    i = args.index("--only"); only = set(args[i + 1].split(",")); args = args[:i] + args[i + 2:]
agent, which, scope = args[0], args[1], args[2]
conc = int(args[3]) if len(args) > 3 else 4
USER = "haiku"
old, new = json.load(open(S / "tib/selection.json")), json.load(open(S / "combo_rep/selection_rep.json"))
allt = {t["id"]: t for t in load_tasks()}
tasks = []
for sc, sel, base, seeded in (("old", old, S / "tib", S / "tib/seeded/probes.json"), ("new", new, S / "combo_rep", S / "combo_rep/probes_seeded.json")):
    if scope not in (sc, "all"):
        continue
    if which == "real":
        tasks += [allt[i] for f in sel["test_families"] for i in sel["real_tasks"][f]]
    elif which in ("doc", "seed", "pseudo"):   # Exp 8 probes written by Opus (gen_claude.py), old families only
        if sc == "old":
            tasks += json.load(open(HERE / f"probes_{which}.json"))
    else:
        tasks += json.load(open(seeded))
if only:
    tasks = [t for t in tasks if t["id"] in only]
out = HERE / "runs" / f"{which}_{agent}.jsonl"
md = HERE / "runs" / "msgs" / f"{which}_{agent}"; md.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)["task_id"] for l in open(out) if json.loads(l).get("reward") is not None} if out.exists() else set()
todo = [t for t in tasks if t["id"] not in done]
print(f"{which} {agent}: {len(tasks)} tasks, {len(done)} done, {len(todo)} to run", flush=True)
cfg = TextRunConfig(domain="indian_banking", agent="llm_agent", llm_agent=f"openai/agent|{agent}",
                    llm_args_agent={"api_base": "http://127.0.0.1:8097/v1", "api_key": "x", "temperature": 0.0},
                    user="user_simulator", llm_user=f"openai/user|{USER}",
                    llm_args_user={"api_base": "http://127.0.0.1:8097/v1", "api_key": "x", "temperature": 0.0},
                    max_steps=120, max_errors=10, seed=300)


def one(t):
    t0 = time.time()
    try:
        res = run_simulation(build_text_orchestrator(cfg, Task.model_validate(t), seed=300))
        ri = res.reward_info
        rec = {"task_id": t["id"], "agent": agent, "set": which, "reward": float(ri.reward) if ri else None,
               "termination": str(res.termination_reason), "n_msgs": len(res.messages), "sec": round(time.time() - t0, 1)}
        (md / f"{t['id']}.json").write_text(json.dumps([m.model_dump(mode="json") for m in res.messages]))
    except Exception as e:
        rec = {"task_id": t["id"], "agent": agent, "set": which, "reward": None, "error": f"{type(e).__name__}: {str(e)[:300]}", "sec": round(time.time() - t0, 1)}
        traceback.print_exc()
    return rec


with ThreadPoolExecutor(conc) as ex, open(out, "a") as f:
    for i, fu in enumerate(as_completed([ex.submit(one, t) for t in todo])):
        r = fu.result(); f.write(json.dumps(r) + "\n"); f.flush()
        print(i + 1, r["task_id"], r.get("reward"), r.get("termination", r.get("error")), r["sec"], flush=True)
