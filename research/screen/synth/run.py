"""Run agent episodes on the synthetic cases of one generator through the local Vertex proxy (PREREG.md).

  python run.py <generator: haiku|sonnet|opus|example> <agent: lite|flash|g38> [concurrency] [--only uid,uid]

One line per finished episode in runs/<generator>_<agent>.jsonl (resumable), full messages in runs/msgs/.
Spending is capped by the proxy (cap_usd.txt), not here.
"""
import json, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["TAU2_AGENT_INSTRUCTION_FILE"] = str(HERE / "agent_instruction.txt")
V = os.environ.get("SYNTH_V", "")
os.environ["TAU2_USER_SIM_GUIDELINES_FILE"] = str(HERE / ("user_guidelines_v2.md" if V == "v2" else "user_guidelines.md"))
import engine, dataset
from tau2.registry import registry
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.runner import build_text_orchestrator, run_simulation

PROXY = "http://127.0.0.1:8094/v1"
AGENTS = {"lite": "gemini-2.5-flash-lite", "flash": "gemini-2.5-flash", "g38": "gemini-3.8-flash"}
USER_MODEL = "gemini-2.5-flash"

args = sys.argv[1:]
only = None
if "--only" in args:
    i = args.index("--only"); only = set(args[i + 1].split(",")); args = args[:i] + args[i + 2:]
gen, agent = args[0], args[1]
conc = int(args[2]) if len(args) > 2 else 6
if gen == "example":
    spec = json.load(open(HERE / "example_procedure.json"))
    brief ={"procedure_id": "example_library", "structure": {"n_rules": 5, "n_calls": 1, "escalation": True, "verification": True},
             "cases": [{"case_id": "c1", "level": "upfront", "behavior": "cooperative", "cls": "perform"},
                       {"case_id": "c2", "level": "lookup", "behavior": "terse", "cls": "perform"},
                       {"case_id": "c3", "level": "conflicting", "behavior": "pushy", "cls": "refuse"},
                       {"case_id": "c4", "level": "insufficient", "behavior": "cooperative", "cls": "refuse"}]}
    dataset.BRIEFS["example_library"] = brief
    items = list(dataset.cases("example", specs={"example_library": spec}))
else:
    items = list(dataset.cases(gen))
if only:
    items = [it for it in items if it["uid"] in only]
if os.environ.get("SYNTH_ORDER_SEED"):  # g38 runs procedures in a fixed random order (PREREG.md stop rule)
    import random
    pids = sorted({it["pid"] for it in items})
    random.Random(int(os.environ["SYNTH_ORDER_SEED"])).shuffle(pids)
    rank = {p: i for i, p in enumerate(pids)}
    items.sort(key=lambda it: (rank[it["pid"]], it["case"]["case_id"]))
for it in items:
    registry.register_domain(engine.make_env_constructor(it["spec"], it["tables"], it["uid"]), it["uid"])

pre = f"{V}_" if V else ""
out = HERE / "runs" / f"{pre}{gen}_{agent}.jsonl"
msgdir = HERE / "runs" / "msgs" / f"{pre}{gen}_{agent}"
msgdir.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    for l in open(out):
        r = json.loads(l)
        if r.get("reward") is not None:
            done.add(r["task_id"])
todo = [it for it in items if it["uid"] not in done]
print(f"{gen} {agent}: {len(items)} cases, {len(done)} done, {len(todo)} to run", flush=True)


def one(it):
    t0 = time.time()
    cfg = TextRunConfig(domain=it["uid"], agent="llm_agent", llm_agent=f"openai/agent|{AGENTS[agent]}",
                        llm_args_agent={"api_base": PROXY, "api_key": "proxy", "temperature": 0.0, "max_tokens": 8192},
                        user="user_simulator", llm_user=f"openai/user|{USER_MODEL}",
                        llm_args_user={"api_base": PROXY, "api_key": "proxy", "temperature": 0.0, "max_tokens": 4096},
                        max_steps=120, max_errors=10, seed=300)
    try:
        task = Task.model_validate(it["task"])
        res = run_simulation(build_text_orchestrator(cfg, task, seed=300))
        ri = res.reward_info
        rec = {"task_id": it["uid"], "gen": gen, "agent": agent, "reward": float(ri.reward) if ri else None,
               "termination": str(res.termination_reason), "n_msgs": len(res.messages), "sec": round(time.time() - t0, 1),
               "db_ok": bool(ri.db_check.db_match) if ri and ri.db_check else None,
               "actions_ok": [bool(a.action_match) for a in (ri.action_checks or [])] if ri else None}
        (msgdir / f"{it['uid']}.json").write_text(json.dumps([m.model_dump(mode="json") for m in res.messages]))
    except Exception as e:
        rec = {"task_id": it["uid"], "gen": gen, "agent": agent, "reward": None, "error": f"{type(e).__name__}: {str(e)[:300]}",
               "sec": round(time.time() - t0, 1)}
        traceback.print_exc()
    return rec


with ThreadPoolExecutor(conc) as ex, open(out, "a") as f:
    futs = [ex.submit(one, it) for it in todo]
    for i, fu in enumerate(as_completed(futs)):
        rec = fu.result()
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(i + 1, rec["task_id"], rec.get("reward"), rec.get("termination", rec.get("error")), rec["sec"], flush=True)
