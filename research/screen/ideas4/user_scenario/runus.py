"""user_scenario run, tau2 only. usage: runus.py [debug]"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, importlib.util
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric"); sys.path.insert(0, S + "/ideas3/conditions")
debug = len(sys.argv) > 1
_argv = sys.argv; sys.argv = ["x", "main"]
spec = importlib.util.spec_from_file_location("combo_run", S + "/ideas3/combo/run.py"); C0 = importlib.util.module_from_spec(spec)
sys.modules["combo_run"] = C0; spec.loader.exec_module(C0); sys.argv = _argv
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import harness as H
from cpulock import cpu_lock
from ladder import Ladder
HERE = Path(__file__).parent
SCN = pd.read_csv(HERE / "scn_tau2.csv").set_index("case_id")
CMAP = json.load(open(HERE / "clause_map.json"))
PROCS = C0.PROC["tau2"]
K_SHRINK = 3.0

def clause_stats(train_ids, y):
    """shrunk mean difficulty per clause from training tasks"""
    g = float(np.mean(y)); sums, cnt = {}, {}
    for i, b in zip(train_ids, y):
        for c in CMAP.get(i, []): sums[c] = sums.get(c, 0) + b; cnt[c] = cnt.get(c, 0) + 1
    return g, {c: (sums[c] + K_SHRINK * g) / (cnt[c] + K_SHRINK) for c in sums}
def clause_feat(ids, g, st):
    out = []
    for i in ids:
        v = [st[c] for c in CMAP.get(i, []) if c in st]
        out.append([np.mean(v), np.max(v)] if v else [g, g])
    return np.array(out)
def clause_table(data, tr):
    """OOF features for training tasks (5 inner folds grouped by procedure), full-train features for test tasks"""
    tr = list(tr); y = data.train_items.loc[tr, "b"].to_numpy(float); z = np.zeros((len(tr), 2))
    grp = PROCS.loc[tr].to_numpy()
    for a_, b_ in GroupKFold(5).split(tr, groups=grp):
        g, st = clause_stats([tr[i] for i in a_], y[a_]); z[b_] = clause_feat([tr[i] for i in b_], g, st)
    te = list(data.test_tasks); g, st = clause_stats(tr, y)
    allz = np.vstack([z, clause_feat(te, g, st)])
    return pd.DataFrame(allz, index=tr + te, columns=["cls_mean", "cls_max"])

class GroupedCls(C0.Grouped):
    def __init__(self, name, fams):
        super().__init__(name, "tau2", [f for f in fams if f != "CLS"]); self.want_cls = "CLS" in fams
        self.tabs["SCN"] = SCN
        if self.want_cls: self.fams = self.fams + ["CLS"]
    def _mat(self, ids, z=None):
        blocks = []
        for f in self.fams:
            blocks.append(self._clst.loc[ids].to_numpy(float) if f == "CLS" else self.tabs[f].loc[ids].to_numpy(float))
        return blocks
    def fit(self, data, tr):
        if self.want_cls: self._clst = clause_table(data, tr)
        super().fit(data, tr)

class LadCls(Ladder):
    def __init__(self, base_extra, with_cls, name):
        super().__init__(2, C0.TYPES, C0.GLOB, C0.static_tables("tau2")["RUB"], extra_table=base_extra, name=name)
        self.base_extra, self.with_cls = base_extra, with_cls
    def fit(self, data, tr):
        if self.with_cls:
            ct = clause_table(data, tr)
            self.extra = self.base_extra.join(ct, how="left").fillna(0.0)
        super().fit(data, tr)

def make_preds(src, M, bench):
    t = C0.static_tables("tau2")
    ex = pd.concat([t["ADELE"], t["PLAN"]], axis=1)
    exs = pd.concat([ex, SCN], axis=1)
    return {"scn_adele": GroupedCls("scn_adele", ["ADELE", "SCN"]),
            "cls_adele": GroupedCls("cls_adele", ["ADELE", "CLS"]),
            "clad_scn": LadCls(exs, False, "clad_scn"),
            "clad_scn_cls": LadCls(exs, True, "clad_scn_cls")}
DESC = {"scn_adele": "grouped ridge ADeLe + scenario-behaviour features (LLM, L0)", "cls_adele": "grouped ridge ADeLe + fold-learned clause difficulty mean/max (L0)",
        "clad_scn": "c_lad (ladder + ADeLe + PLAN) + scenario features", "clad_scn_cls": "c_lad + scenario + clause features"}
if __name__ == "__main__":
    out = HERE / ("out_debug" if debug else "out")
    with cpu_lock("user_scenario"):
        H.run_all(out, make_preds, benches=["tau2"], schemes=["new_domain"] if debug else ["new_procedures", "new_domain"])
    print("run done", out)
