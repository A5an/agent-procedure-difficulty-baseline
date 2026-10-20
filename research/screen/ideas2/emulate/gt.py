"""Ground truth by code for a (possibly synthetic) SOPBench case, using the env's own classes."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, copy, json, re
SOP = ZR.DATA + "/SOPBench-d2622008"
sys.path.insert(0, SOP)
os.chdir(SOP)
from env.variables import domain_keys, domain_assistant_keys
from env.task import task_default_dep_full, task_initializer
_dd = {}
def defaults(dom):
    if dom not in _dd:
        _dd[dom] = task_default_dep_full(dom, "full", "structured", dependency_verb_dep_orig=True)
    return _dd[dom]

def leaves(d, out):
    if not d: return
    if d[0] == "single": out.append(d[1].replace("not ", "")); return
    for x in d[1]: leaves(x, out)

def leaf_results(dom, case, db=None, goal=None):
    """Evaluate every leaf of case['constraints'] on db. Returns (label, {leaf_key: bool}) ."""
    dep_innate, default_dep, _ = defaults(dom)
    dep_full = copy.deepcopy(default_dep); goal = goal or case["user_goal"]; dep_full[goal] = case["constraints"]
    data = copy.deepcopy(db if db is not None else case["initial_database"])
    st = domain_keys[dom + "_strict"](data, dep_innate, dep_full, case["constraint_parameters"])
    uk = dict(case["user_known"])
    # prerequisite steps that a correct agent would do: login, admin password (credential argument names come from the method signature)
    import inspect
    try:
        if "username" in uk and hasattr(st, "login_user"):
            params = [p for p in inspect.signature(st.login_user).parameters]
            kw = {k: uk[k] for k in params if k in uk}
            if len(kw) == len(params) or ("username" in kw and len(kw) > 1):
                ok = st.login_user(**kw)
                ok = ok[0] if isinstance(ok, tuple) else ok
                if not ok:
                    tr = st.state_tracker
                    if hasattr(tr, "remove_logged_in_user"): tr.remove_logged_in_user(uk["username"])
                    elif hasattr(tr, "set_logout_user"): tr.set_logout_user()
            if "admin_password" in uk and hasattr(st, "authenticate_admin_password"):
                ok = st.authenticate_admin_password(username=uk["username"], admin_password=uk["admin_password"])
                if not ok: st.state_tracker.prev_auth_admin_pass = False
    except Exception:
        pass
    ev = st.evaluation_get_dependency_evaluator()
    _orig = ev._single
    def _safe(func, pm, **kw):
        try: return _orig(func, pm, **kw)
        except Exception: return False   # an exception while evaluating a leaf counts as the leaf failing
    ev._single = _safe
    res = {}
    def walk(d):
        if not d: return
        if d[0] == "single":
            key = d[1] + json.dumps(d[2], sort_keys=True)
            res[key] = bool(ev._process(d, **uk)); return
        for x in d[1]: walk(x)
    walk(case["constraints"])
    label = bool(ev._process(case["constraints"], **uk)) if case["constraints"] else True
    return label, res

def load_check(dom, case, db=None, goal=None):
    """Does the case load into the environment and does the goal tool run? returns (ok, err)"""
    try:
        dep_innate, default_dep, descr = defaults(dom)
        c = copy.deepcopy(case)
        if db is not None: c["initial_database"] = copy.deepcopy(db)
        c["user_goal"] = goal or case.get("user_goal")
        sysd, ui, ai, ti = task_initializer(dom, c, dep_innate, default_dep, descr, None, "prompt", False, "structured")
        return True, None
    except Exception as e:
        return False, repr(e)[:200]

def policy_prompt(dom, case, goal=None):
    case = dict(case); case["user_goal"] = goal or case.get("user_goal")
    dep_innate, default_dep, descr = defaults(dom)
    sysd, ui, ai, ti = task_initializer(dom, case, dep_innate, default_dep, descr, None, "prompt", False, "structured")
    return ai["instructions"], ai["tools"]
