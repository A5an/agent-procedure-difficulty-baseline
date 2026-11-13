"""Episodes for E1 (probes of the new conditions) and E2 (the existing seeded probes with a second user simulator), a
copy of ../combo_rep/run_rep.py with this folder's proxy (port 8095) and selectable user simulator.

  <tau venv python> run_seedk.py <agent: g38|flash> <set: k1|k5|foreign|seeded> <user: flash|pro> [concurrency] [family ...]
k1|k5|foreign: probes/<set>/<family>.json of this folder, outputs runs/<set>_<agent>.jsonl here.
seeded (E2): the existing k=2 seeded probes (../tib/seeded/probes.json, ../combo_rep/probes_seeded.json), outputs
../sim2/runs/seeded_<user>_<agent>.jsonl. Default families: the 5 pilot families. Resumable; spending capped by the proxy."""
import json, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tib"))
from common import AGENTS, family

from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.runner import build_text_orchestrator, run_simulation

PROXY = "http://127.0.0.1:8095/v1"
USERS = {"flash": "gemini-2.5-flash", "pro": "gemini-2.5-pro"}
agent, which, user = sys.argv[1], sys.argv[2], sys.argv[3]
conc = int(sys.argv[4]) if len(sys.argv) > 4 else 6
SEL = json.load(open(HERE / "selection_seedk.json"))
fams = set(sys.argv[5:] or SEL["pilot_families"])
if which == "seeded":
    tasks = json.load(open(HERE.parent / "tib" / "seeded" / "probes.json")) + json.load(open(HERE.parent / "combo_rep" / "probes_seeded.json"))
    base = HERE.parent / "sim2" / "runs"; name = f"seeded_{user}_{agent}"
else:
    tasks = [t for f in sorted(fams) if (HERE / "probes" / which / f"{f}.json").exists() for t in json.load(open(HERE / "probes" / which / f"{f}.json"))]
    base = HERE / "runs"; name = f"{which}_{agent}" + ("" if user == "flash" else f"_{user}")
tasks = [t for t in tasks if family(t) in fams]
out = base / f"{name}.jsonl"; msgdir = base / "msgs" / name
msgdir.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    for l in open(out):
        r = json.loads(l)
        if r.get("reward") is not None: done.add(r["task_id"])
todo = [t for t in tasks if t["id"] not in done]
print(f"{name}: {len(tasks)} tasks, {len(done)} done, {len(todo)} to run", flush=True)

cfg = TextRunConfig(domain="indian_banking", agent="llm_agent",
                    llm_agent=f"openai/agent|{AGENTS[agent]}",
                    llm_args_agent={"api_base": PROXY, "api_key": "proxy", "temperature": 0.0, "max_tokens": 8192},
                    user="user_simulator", llm_user=f"openai/user|{USERS[user]}",
                    llm_args_user={"api_base": PROXY, "api_key": "proxy", "temperature": 0.0, "max_tokens": 8192},
                    max_steps=200, max_errors=10, seed=300)


def one(t):
    t0 = time.time()
    try:
        task = Task.model_validate(t)
        res = run_simulation(build_text_orchestrator(cfg, task, seed=300))
        ri = res.reward_info
        rec = {"task_id": t["id"], "agent": agent, "set": which, "user": user, "reward": float(ri.reward) if ri else None,
               "termination": str(res.termination_reason), "n_msgs": len(res.messages), "sec": round(time.time() - t0, 1),
               "db_ok": bool(ri.db_check.db_match) if ri and ri.db_check else None,
               "actions_ok": [bool(a.action_match) for a in (ri.action_checks or [])] if ri else None}
        (msgdir / f"{t['id']}.json").write_text(json.dumps([m.model_dump(mode="json") for m in res.messages]))
    except Exception as e:
        rec = {"task_id": t["id"], "agent": agent, "set": which, "user": user, "reward": None,
               "error": f"{type(e).__name__}: {str(e)[:300]}", "sec": round(time.time() - t0, 1)}
        traceback.print_exc()
    return rec


with ThreadPoolExecutor(conc) as ex, open(out, "a") as f:
    futs = [ex.submit(one, t) for t in todo]
    for i, fu in enumerate(as_completed(futs)):
        rec = fu.result()
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(i + 1, rec["task_id"], rec.get("reward"), rec.get("termination", rec.get("error")), rec["sec"], flush=True)
