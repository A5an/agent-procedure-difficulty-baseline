"""session creation, prompt building. import-safe helpers."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, copy, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sop_tool as T
COMP = ZR.SCREEN + "/ideas2/compile/"

def required_tools(code, goal):  # identical to harness/run.py
    pos = [m.start() for m in re.finditer(r"tools\.%s\b" % re.escape(goal), code)]
    cut = pos[0] if pos else len(code)
    names = []
    for m in re.finditer(r"tools\.(\w+)", code[:cut]):
        if m.group(1) not in names and m.group(1) != goal: names.append(m.group(1))
    return names

_meta = {}
def guard_for(case_id, goal):
    if not _meta:
        _meta["units"] = {r["case_id"]: r["unit"] for r in json.load(open(COMP + "results_A.json"))}
        _meta["progs"] = json.load(open(COMP + "programs_t0.json"))
    return required_tools(_meta["progs"][_meta["units"][case_id]]["code"], goal)

TEMPLATE = open(os.path.join(HERE, os.environ.get("CC_TEMPLATE", "prompt_template.txt"))).read()

def build_prompt(case_id, session_dir, source="real"):
    t = T._load_case(case_id, source)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        ds, ai = T.build(t)
    tools = json.dumps([a for a in ai["tools"]], indent=1)
    return TEMPLATE.format(system=ai["instructions"], tools=tools,
                           cmd=f"python3 {HERE}/sop_tool.py {session_dir} call <tool_name> '<json arguments>'",
                           customer=t["user_prompt"])

def make_session(case_id, session_dir, source="real"):
    os.makedirs(session_dir, exist_ok=True)
    meta = {"case_id": case_id, "guard": None, "source": source}
    json.dump(meta, open(os.path.join(session_dir, "meta.json"), "w"))
    open(os.path.join(session_dir, "log.jsonl"), "w").close()
