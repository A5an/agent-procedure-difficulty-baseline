"""Code-plan variants, built like ideas3/combo/run.py. usage: run.py main [debug]"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, itertools
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric")
sys.path.insert(0, S + "/ideas3/conditions")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.model_selection import GroupKFold
import harness as H
from cpulock import cpu_lock
from ladder import Ladder
import prompts as P
I3 = Path(S + "/ideas3"); HERE = Path(S + "/ideas4/code_plan")
REPO = H.REPO
mode = sys.argv[1]; debug = len(sys.argv) > 2
GRID = [1.0, 10.0, 100.0, 1000.0, 10000.0]
RUBF = Path(S + "/rubric/features")

def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
def static_tables(b):
    ad = tab(REPO / "data" / b / "features" / "adele.csv")
    rub = tab(RUBF / f"rub_proc_{b}.csv")
    plan = tab(I3 / "plan_tools/features" / f"plan3_{b}.csv")
    t = {"ADELE": ad, "RUB": rub, "PLAN": plan, "CP": tab(HERE / f"cp_{b}.csv") if (HERE / f"cp_{b}.csv").exists() else None}
    if b == "sopbench":
        t["CMP"] = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]]
        t["CMPP"] = tab(HERE / "cmpplus_sopbench.csv")
    return t
PROC = {b: pd.read_csv(REPO / "data" / b / "tasks.csv").set_index("case_id").procedure_id for b in ["sopbench", "tau2"]}

class Grouped:
    def __init__(self, name, bench, fams):
        self.name, self.bench, self.fams = name, bench, fams
        self.tabs = static_tables(bench); self._predicted_difficulties = {}
        self.cond = "COND" in fams
        if self.cond:
            import cm
            self.cm = cm; self.pr = cm.Pred("l0", "l0")
    def _mat(self, ids, z=None):
        blocks = []
        for f in self.fams:
            blocks.append(np.asarray(z, float)[:, None] if f == "COND" else self.tabs[f].loc[ids].to_numpy(float))
        return blocks
    def fit(self, data, tr):
        self._predicted_difficulties = {}
        tr = list(tr); y = data.train_items.loc[tr, "b"].to_numpy(float)
        z = None
        if self.cond:
            grp = PROC[self.bench].loc[tr].to_numpy()
            z = np.zeros(len(tr))
            for a_, b_ in GroupKFold(5).split(tr, groups=grp):
                cmi = self.cm.CondModel([tr[i] for i in a_])
                z[b_] = self.pr._z(cmi, [tr[i] for i in b_])
            self.cm_full = self.cm.CondModel(tr)
        blocks = self._mat(tr, z)
        self.sizes = [bk.shape[1] for bk in blocks]
        X = np.hstack(blocks); grp = PROC[self.bench].loc[tr].to_numpy()
        folds = []
        for a_, b_ in GroupKFold(5).split(X, groups=grp):
            m, s = X[a_].mean(0), X[a_].std(0); s = np.where(s < 1e-9, np.inf, s)
            Za, Zb = (X[a_] - m) / s, (X[b_] - m) / s
            yb = y[a_].mean()
            folds.append((Za.T @ Za, Za.T @ (y[a_] - yb), Zb, y[b_], yb))
        best, bal = np.inf, None
        for combo in itertools.product(GRID, repeat=len(self.fams)):
            sc = np.concatenate([np.full(n, 1 / np.sqrt(a)) for n, a in zip(self.sizes, combo)])
            mse = 0.0
            for XtX, Xty, Zb, yv, yb in folds:
                A = XtX * np.outer(sc, sc) + np.eye(len(sc))
                w = np.linalg.solve(A, sc * Xty)
                mse += np.mean((Zb @ (sc * w) + yb - yv) ** 2)
            if mse < best: best, bal = mse, combo
        self.alphas = dict(zip(self.fams, bal))
        sc = np.concatenate([np.full(n, 1 / np.sqrt(a)) for n, a in zip(self.sizes, bal)])
        self.m, self.s = X.mean(0), X.std(0); self.s = np.where(self.s < 1e-9, np.inf, self.s)
        Z = (X - self.m) / self.s
        self.ybar = y.mean(); self.sc = sc
        A = (Z.T @ Z) * np.outer(sc, sc) + np.eye(len(sc))
        self.w = np.linalg.solve(A, sc * (Z.T @ (y - self.ybar)))
        te = list(data.test_tasks)
        zt = None
        if self.cond:
            zt = self.pr._z(self.cm_full, te)
        Xt = np.hstack(self._mat(te, zt))
        pred = ((Xt - self.m) / self.s) @ (self.sc * self.w) + self.ybar
        self._predicted_difficulties = dict(zip(te, pred))
        print(f"   {self.name} {self.bench} alphas {self.alphas}", flush=True)
    def predict_probability(self, data, agent, task):
        th = data.train_abilities.loc[agent]; th = float(th.iloc[0]) if hasattr(th, "iloc") else float(th)
        return float(np.clip(expit(th - self._predicted_difficulties[task]), 1e-4, 1 - 1e-4))

TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    t = static_tables(bench)
    ex = pd.concat([t["ADELE"], t["PLAN"]] + ([t["CMP"]] if sop else []), axis=1)
    if t["CP"] is None: return {"c_lad_cmpp": Ladder(2, TYPES, GLOB, t["RUB"], extra_table=pd.concat([t["ADELE"], t["PLAN"], t["CMPP"]], axis=1), name="c_lad_cmpp")}
    ex_cp = pd.concat([t["ADELE"], t["PLAN"]] + ([t["CMP"]] if sop else []) + [t["CP"]], axis=1)
    ex_pp = pd.concat([t["ADELE"], t["PLAN"]] + ([t["CMPP"]] if sop else []), axis=1)
    return {"cp_adele": Grouped("cp_adele", bench, ["ADELE", "CP"]),
            "c_lad_cp": Ladder(2, TYPES, GLOB, t["RUB"], extra_table=ex_cp, name="c_lad_cp"),
            "c_lad_cmpp": Ladder(2, TYPES, GLOB, t["RUB"], extra_table=ex_pp, name="c_lad_cmpp")}
DESC = {"cp_adele": "L1: grouped ridge ADELE + code-plan AST features", "c_lad_cp": "c_lad + code-plan AST features",
        "c_lad_cmpp": "c_lad with CMP replaced by CMP+ (sopbench AST features of compiled programs, L0); tau2 identical to c_lad"}
COMBOS = [("cp_adele", "adele"), ("c_lad_cp", "adele")]
if __name__ == "__main__":
    if mode == "top5":
        H.RB.DATA = I3 / "tiers/data/top5"; H.RB.RESULTS = I3 / "tiers/irt_cache/top5"
    out = HERE / ("out_" + mode + ("_debug" if debug else ""))
    benches = ["tau2"] if mode == "tau2" else ["sopbench"] if mode == "sopL0" else ["sopbench", "tau2"]
    schemes = ["new_procedures", "new_domain"]
    with cpu_lock("code_plan"):
        H.run_all(out, make_preds, benches=benches, schemes=schemes)
    print("run done", out)
