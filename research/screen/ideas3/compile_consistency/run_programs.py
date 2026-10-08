"""Run the 5 compiled programs of every unit on every synthetic scenario (deterministic code, no LLM, no real records).
Output: runs.json  {unit: {"scen": n, "progs": [ [ {goal_called, tools, calls, err, n_tool_err} per scenario ] per program ]}}"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, copy, signal
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, ZR.SCREEN + "/ideas")
import sop, common
from env.variables import domain_keys
from multiprocessing import Pool
C2 = ZR.SCREEN + "/ideas2/compile/"
FILES = [C2 + "programs_t0.json", C2 + "programs_t1_s1.json", C2 + "programs_t1_s2.json", HERE + "/programs_t1_s3.json", HERE + "/programs_t1_s4.json"]
class Timeout(Exception): pass
def _alarm(*a): raise Timeout()
def try_eval(x):
    try: return eval(x)
    except Exception: return x
class Tools:
    def __init__(s, ds, allowed): s.ds, s.allowed, s.log = ds, allowed, []
    def __getattr__(s, name):
        if name.startswith("__"): raise AttributeError(name)
        def call(*args, **kw):
            if args: return "Error: positional arguments not supported"
            if len(s.log) >= 30: raise Timeout()
            if name not in s.allowed or not hasattr(s.ds, name):
                s.log.append((name, kw, "Error: not found", True)); return "Error: Tool not found."
            try:
                raw = getattr(s.ds, name)(**kw)
                if isinstance(raw, tuple): raw = raw[1]
                bad = False
            except Exception as e:
                raw = f"{e.__class__.__name__}: {str(e)}"; bad = True
            r = try_eval(str(raw))
            s.log.append((name, kw, str(raw)[:80], bad or (isinstance(r, str) and ("Error" in r[:40]))))
            return r
        return call
def run_one(code, params, t, db, allowed):
    inn, full, descr = sop.defaults(t["domain"])
    ds = domain_keys[t["domain"]](copy.deepcopy(db), inn, t["constraint_parameters"])
    tools = Tools(ds, allowed); err = None
    try:
        g = {"__builtins__": __builtins__}; exec(code, g)
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(3)
        try: g["handle"](copy.deepcopy(params), tools)
        finally: signal.alarm(0)
    except Timeout: err = "timeout"
    except BaseException as e: err = e.__class__.__name__
    goal = t["user_goal"]
    names = [l[0] for l in tools.log]
    return {"goal": goal in names, "tools": sorted(set(names)), "calls": sorted(set(l[0] + json.dumps(l[1], sort_keys=True, default=str) for l in tools.log)),
            "err": err, "n_calls": len(tools.log), "n_tool_err": sum(1 for l in tools.log if l[3]),
            "tool_err_names": sorted(set(l[0] for l in tools.log if l[3]))}
def work(a):
    k, t, scen, codes = a
    _, ai = sop.agent_view(t); allowed = {x["function"]["name"] for x in ai["tools"]}
    out = []
    for code in codes:
        if code is None: out.append(None); continue
        out.append([run_one(code, c["user_known"], t, c["initial_database"], allowed) for c in scen])
    return k, {"scen": len(scen), "progs": out}
if __name__ == "__main__":
    from cpulock import cpu_lock
    syn = json.load(open(HERE + "/synthetic.json"))
    progs = [json.load(open(f)) for f in FILES]
    u = {}
    for t in sop.load_cases(): u.setdefault(common.key_of(t), t)
    jobs = [(k, u[k], syn[k]["scen"][:10], [(p.get(k) or {}).get("code") for p in progs]) for k in u if syn.get(k, {}).get("scen")]
    res = {}
    with cpu_lock("compile_consistency"):
        with Pool(4) as pool:
            for i, (k, r) in enumerate(pool.imap_unordered(work, jobs, chunksize=2)):
                res[k] = r
                if i % 50 == 0: print(i, flush=True)
    json.dump(res, open(HERE + "/runs.json", "w"))
    print("done", len(res))
