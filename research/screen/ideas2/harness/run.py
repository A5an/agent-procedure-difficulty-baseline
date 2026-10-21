import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os, sys, json, csv, copy, re, time, threading, argparse, traceback, io, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); SOP = os.path.join(HERE, "sop")
sys.path.insert(0, SOP); os.chdir(SOP)
from concurrent.futures import ThreadPoolExecutor
import swarm.core as core
from swarm.core import *
from swarm import vclient
from swarm.vclient import VClient, ctx, BudgetExceeded
import run_simulation as rs
from env.task import task_default_dep_full
from env.evaluator import evaluator_function_directed_graph
from run_evaluation import try_eval

COMP = ZR.SCREEN + "/ideas2/compile/"
OUT = os.path.join(HERE, "episodes"); os.makedirs(OUT, exist_ok=True)

# ---- guard (H2) ----
_orig_htc = core.Swarm.handle_tool_calls
def guarded_htc(self, tool_calls, functions, context_variables, debug):
    g = getattr(ctx, "guard", None)
    if g is None: return _orig_htc(self, tool_calls, functions, context_variables, debug)
    resp = Response(messages=[], agent=None, context_variables={})
    for tc in tool_calls:
        name = tc.function.name
        if g["enabled"] and name == g["goal"]:
            missing = [t for t in g["required"] if t not in g["called"]]
            if missing:
                msg = ("Error: action not executed. The policy for this action requires checks that have not been made in this conversation: "
                       + ", ".join(missing) + ". Call these verification tools first, then retry only if every condition is satisfied.")
                g["blocks"].append({"missing": missing, "args": tc.function.arguments, "n_called": len(g["called"])})
                resp.messages.append({"role": "tool", "tool_call_id": tc.id, "tool_name": name, "content": msg, "_blocked": True})
                continue
        part = _orig_htc(self, [tc], functions, context_variables, debug)
        g["called"].add(name)
        resp.messages.extend(part.messages)
    return resp
core.Swarm.handle_tool_calls = guarded_htc

def required_tools(code, goal):
    pos = [m.start() for m in re.finditer(r"tools\.%s\b" % re.escape(goal), code)]
    cut = pos[0] if pos else len(code)
    names = []
    for m in re.finditer(r"tools\.(\w+)", code[:cut]):
        if m.group(1) not in names and m.group(1) != goal: names.append(m.group(1))
    return names

def load_meta():
    units = {r["case_id"]: r["unit"] for r in json.load(open(COMP + "results_A.json"))}
    progs = json.load(open(COMP + "programs_t0.json"))
    return units, progs

def load_tasks():
    tasks = {}
    for d in ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]:
        td = json.load(open(f"{SOP}/data/{d}_tasks.json"))
        for goal, arr in td.items():
            for i, t in enumerate(arr):
                t = copy.deepcopy(t); t["user_goal"] = goal; t["domain"] = d
                tasks[f"{d}/{goal}/{i}"] = t
    return tasks

class A: pass
def make_args(domain, scaffold):
    a = A()
    a.tool_list = "full"; a.scaffold = scaffold; a.default_constraint_option = "full"; a.constraint_descr_format = "structured"
    a.constraint_hint = a.constraint_verdict = a.action_order = False
    a.env_mode = "prompt"; a.shuffle_func = False; a.user_model = None; a.domain = domain
    a.max_num_turns = 20; a.max_num_actions = 10; a.print_conv = False; a.tool_call_mode = "fc"
    return a

_DEPS = {}
def deps(domain):
    if domain not in _DEPS: _DEPS[domain] = task_default_dep_full(domain, "full", "structured", dependency_verb_dep_orig=True)
    return _DEPS[domain]

def run_episode(model, kind, region, arm, case_id, task, units, progs, max_tokens, temp=0.0):
    scaffold = "h1" if arm == "H1" else "none"
    a = make_args(task["domain"], scaffold)
    dep_in, dep_full, dep_descr = deps(task["domain"])
    client = VClient(model, kind=kind, region=region)
    ag = Agent(name="domain assistant", client=client, temperature=temp, top_p=0.01 if kind == "gemini" else 1.0, max_tokens=max_tokens,
               tool_call_mode="fc", instructions="", functions=[])
    user = Agent(name="user", client=None, default_response="", response_repeat=True)
    ctx.tag = (model, arm, case_id); ctx.usage = []; ctx.sigs = {}
    guard = None
    if arm == "H2":
        unit = units[case_id]; req = required_tools(progs[unit]["code"], task["user_goal"])
        guard = {"enabled": True, "goal": task["user_goal"], "required": req, "called": set(), "blocks": []}
    ctx.guard = guard
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = rs.run_task_simulation(a, copy.deepcopy(task), dep_in, dep_full, dep_descr, ag, user)
    return res, guard

def strip_blocked(interaction):
    """remove blocked target attempts (assistant call + guard error reply): they never touched the environment"""
    out = []; i = 0
    while i < len(interaction):
        m = interaction[i]
        if m.get("tool_calls") and i + 1 < len(interaction) and interaction[i + 1].get("_blocked"):
            i += 2; continue
        out.append(m); i += 1
    return out

def evaluate(task, domain, interaction, database, strip=False):
    if strip: interaction = strip_blocked(interaction)
    inter = copy.deepcopy(interaction)
    func_calls = []
    for i in range(len(inter) - 1):
        if inter[i].get("tool_calls"):
            for tc in list(inter[i]["tool_calls"]):
                if tc["function"]["name"].lower() in ["n/a", "na", "none", "null"]: inter[i]["tool_calls"].remove(tc)
            if len(inter[i]["tool_calls"]) > 0:
                func_calls.append({"tool_name": inter[i + 1]["tool_name"], "arguments": try_eval(inter[i]["tool_calls"][0]["function"]["arguments"]),
                                   "content": try_eval(inter[i + 1]["content"])})
    return evaluator_function_directed_graph(domain_str=domain, task=task, log_msg_fcall=inter, func_calls=func_calls,
                                             results={"final_database": database}, default_constraint_option="full")

def episode_path(model, arm): return os.path.join(OUT, f"{model}__{arm}.jsonl")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--arm"); ap.add_argument("--kind", default="gemini"); ap.add_argument("--region", default="global")
    ap.add_argument("--cases", default=os.path.join(HERE, "sample.csv")); ap.add_argument("--only", default=None, help="comma list of case ids")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--domain", default=None)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--cap", type=float, default=None, help="USD cap for this model"); ap.add_argument("--total_cap", type=float, default=45.0)
    ap.add_argument("--out_suffix", default="")
    args = ap.parse_args()
    vclient.CAPS["total"] = args.total_cap
    if args.cap: vclient.CAPS[args.model] = args.cap
    vclient.load_ledger()
    rows = list(csv.DictReader(open(args.cases)))
    ids = [r["case_id"] for r in rows]
    if args.domain: ids = [c for c in ids if c.startswith(args.domain + "/")]
    if args.only: ids = args.only.split(",")
    if args.limit: ids = ids[:args.limit]
    tasks = load_tasks(); units, progs = load_meta()
    path = episode_path(args.model, args.arm + args.out_suffix)
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if not r.get("error"): done.add(r["case_id"])
    todo = [c for c in ids if c not in done]
    print(f"{args.model} {args.arm}: {len(todo)} to run, {len(done)} done; spent {vclient.spent():.2f}", flush=True)
    wlock = threading.Lock(); stop = {"v": False}
    def work(cid):
        if stop["v"]: return
        t0 = time.time(); err = None; res = None; guard = None
        for attempt in range(3):
            try:
                res, guard = run_episode(args.model, args.kind, args.region, args.arm, cid, tasks[cid], units, progs, args.max_tokens, 0.0 if attempt == 0 else 0.7)
                err = None; break
            except BudgetExceeded as e:
                stop["v"] = True; err = "BUDGET " + str(e); break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:300]}"
                ctx.usage = getattr(ctx, "usage", [])
        rec = {"case_id": cid, "model": args.model, "arm": args.arm, "error": err, "secs": round(time.time() - t0, 1),
               "usage": getattr(ctx, "usage", []), "max_tokens": args.max_tokens}
        if res:
            rec["interaction"] = res["interaction"]; rec["database"] = res["database"]
            rec["blocks"] = guard["blocks"] if guard else []
            rec["required"] = guard["required"] if guard else None
            rec["usd"] = sum(u["usd"] for u in rec["usage"])
        with wlock:
            with open(path, "a") as f: f.write(json.dumps(rec, default=str) + "\n")
            print(f"{cid} {'ERR ' + err if err else 'ok'} {rec['secs']}s spent={vclient.spent():.2f}", flush=True)
    with ThreadPoolExecutor(args.workers) as ex: list(ex.map(work, todo))
    print("finished", args.model, args.arm, "spent", round(vclient.spent(), 2), flush=True)

if __name__ == "__main__": main()
