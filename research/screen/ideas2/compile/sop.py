import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, copy, os
SOP=ZR.DATA + "/SOPBench-d2622008"
sys.path.insert(0, SOP); os.chdir(SOP)
from env.task import task_default_dep_full, task_initializer
DOMAINS=["bank","dmv","healthcare","hotel","library","online_market","university"]
def load_cases():
    cases=[]
    for d in DOMAINS:
        td=json.load(open(f"{SOP}/data/{d}_tasks.json"))
        for goal,arr in td.items():
            for i,t in enumerate(arr):
                t=copy.deepcopy(t); t["user_goal"]=goal; t["domain"]=d; t["case_id"]=f"{d}/{goal}/{i}"
                cases.append(t)
    return cases
_dd={}
def defaults(d):
    if d not in _dd: _dd[d]=task_default_dep_full(d,"full","structured",dependency_verb_dep_orig=True)
    return _dd[d]
def agent_view(t):
    """what the agent sees: instructions text and tool list (as in run_simulation, tool_list=full, prompt mode)"""
    d=t["domain"]; inn,full,descr=defaults(d)
    ds,ui,ai,ti=task_initializer(d,copy.deepcopy(t),inn,copy.deepcopy(full),copy.deepcopy(descr),None,"prompt",False,"structured")
    return ds,ai
