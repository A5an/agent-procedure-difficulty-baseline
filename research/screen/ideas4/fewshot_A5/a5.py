"""Stage-A variants (tau2 and SOPBench), training gold only."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
S = ZR.SCREEN
REPO = Path(ZR.REPO)
PT = Path(S + "/ideas3/plan_tools")
EXT = Path(ZR.DATA)
TYPES = json.load(open(PT / "tool_types.json"))
_AF = ["plan_calls", "plan_writes", "plan_reads", "plan_distinct_tools", "plan_rules", "plan_facts", "plan_refuse", "plan_transfer", "plan_ambig"]
TASKS = {b: pd.read_csv(REPO / "data" / b / "tasks.csv").set_index("case_id") for b in ["tau2", "sopbench"]}
PLAN3 = {b: pd.read_csv(PT / f"features/plan3_{b}.csv").set_index("case_id")[_AF] for b in ["tau2", "sopbench"]}
_GOLD = {"tau2": {}, "sopbench": {}}
_m = json.load(open(REPO / "data/tau2/id_map.json"))
for dom in ["airline", "retail", "telecom"]:
    tk = {str(t["id"]): t for t in json.load(open(EXT / f"tau2-domains/tasks/{dom}_tasks.json"))}
    for k, c in _m.items():
        if k.startswith(dom + ":"):
            acts = tk[k.split(":", 1)[1]]["evaluation_criteria"].get("actions") or []
            ag = [a for a in acts if a.get("requestor", "assistant") != "user"]
            _GOLD["tau2"][c] = [len(ag), sum(TYPES.get(f"{dom}/{a['name']}") == "write" for a in ag), len(acts)]
for dom in sorted(TASKS["sopbench"].domain.unique()):
    data = json.load(open(EXT / f"SOPBench-d2622008/data/{dom}_tasks.json"))
    for g, lst in data.items():
        for j, t in enumerate(lst):
            nodes = t["directed_action_graph"]["nodes"]
            act = [n for n in nodes if not isinstance(n, str)]
            _GOLD["sopbench"][f"{dom}/{g}/{j}"] = [len(act), sum(str(n[0]).startswith("internal_check") for n in act), len(nodes)]
assert set(_GOLD["sopbench"]) >= set(TASKS["sopbench"].index), "sop gold map incomplete"
LOOKUPS = []
def gold_of(bench, case, allowed):
    assert case in allowed, f"gold of non-training task {case}"
    LOOKUPS.append((bench, case)); return _GOLD[bench][case]
COLS = ["len", "writes", "all"]
def _cv(X, y, grp):
    m, s = X.mean(0), X.std(0) + 1e-9; Z = (X - m) / s; best, ba = 1e18, 1.0
    for a in [1, 10, 100, 1000]:
        mse = 0
        for i, j in GroupKFold(5).split(Z, groups=grp):
            mse += ((Ridge(alpha=a).fit(Z[i], y[i]).predict(Z[j]) - y[j]) ** 2).mean()
        if mse < best: best, ba = mse, a
    return m, s, ba
def stage_a(bench, tr, te, demean=False):
    tr = list(tr); te = list(te); tr_set = set(tr); assert not (tr_set & set(te))
    X_all = PLAN3[bench]; T = TASKS[bench]
    Y = np.array([gold_of(bench, c, tr_set) for c in tr], float)
    dom = T.domain.loc[tr].to_numpy()
    if demean:
        Y = Y.copy()
        for d in np.unique(dom): Y[dom == d] -= Y[dom == d].mean(0)
    X = X_all.loc[tr].to_numpy(float); grp = T.procedure_id.loc[tr].to_numpy()
    m, s, a = _cv(X, Y, grp)
    oof = np.zeros_like(Y)
    for i, j in GroupKFold(5).split(X, groups=grp):
        oof[j] = Ridge(alpha=a).fit((X[i] - m) / s, Y[i]).predict((X[j] - m) / s)
    pt = Ridge(alpha=a).fit((X - m) / s, Y).predict((X_all.loc[te].to_numpy(float) - m) / s)
    return pd.concat([pd.DataFrame(oof, index=tr, columns=COLS), pd.DataFrame(pt, index=te, columns=COLS)])
