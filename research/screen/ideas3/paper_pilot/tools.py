import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os, sys, json, copy, io, contextlib
SOP=ZR.SCREEN + "/ideas2/harness/sop"
sys.path.insert(0,SOP); os.chdir(SOP)
sys.argv=["x"]
import run_simulation as rs
from env.task import task_default_dep_full
out={}
for d in ["bank","dmv","healthcare","hotel","library","online_market","university"]:
    td=json.load(open(f"{SOP}/data/{d}_tasks.json"))
    goal=list(td)[0]; t=copy.deepcopy(td[goal][0]); t["user_goal"]=goal
    dep=task_default_dep_full(d,"full","structured",dependency_verb_dep_orig=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ds,ui,ai,ti=rs.task_initializer(d,t,*dep,None,"prompt",False,"structured",abl_hint=False,abl_verdict=False,abl_order=False)
    out[d]={"tools":ai["tools"],"instr":ai["instructions"]}
json.dump(out,open(ZR.SCREEN + "/ideas3/paper_pilot/domain_tools.json","w"),indent=1,default=str)
print({d:len(v["tools"]) for d,v in out.items()})
print(out["bank"]["tools"][0]); print(out["bank"]["instr"][:1500])
