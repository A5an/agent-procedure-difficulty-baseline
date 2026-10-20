"""Rubric ladder L2 with one ridge penalty per feature family, chosen by inner 5-fold CV grouped by procedure
(log-loss, coordinate descent: one sweep over families from a common start). Column scaling trick: family f is
multiplied by sqrt(LAM0/lam_f) and fitted with the common penalty LAM0, which equals penalty lam_f on that family."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S + "/rubric")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from ladder import Ladder, _cells
LAM0 = 0.03
GRID = [0.003, 0.03, 0.3, 3.0, 30.0]
START = 0.3
PROC = {b: pd.read_csv(f"{ZR.REPO}/data/{b}/tasks.csv").set_index("case_id").procedure_id
        for b in ["sopbench", "tau2"]}

class FamLadder(Ladder):
    def __init__(self, bench, rub, step_cols, glob_cols, fams, name, per_family=True):
        """fams: {family: DataFrame indexed by case_id}"""
        super().__init__(2, step_cols, glob_cols, rub, extra_table=None, name=name)
        self.bench, self.fams, self.per_family = bench, dict(fams), per_family
        self.fnames = ["STEP", "GLOB"] + list(fams)
    def _blocks(self, ids):
        N = self.table.loc[ids, self.step_cols].to_numpy(float)
        Gs = [self.table.loc[ids, self.glob_cols].to_numpy(float)] + [t.loc[ids].to_numpy(float) for t in self.fams.values()]
        return N, Gs
    def _X(self, ids):
        N, Gs = self._blocks(ids); return N, np.hstack(Gs)
    def _scales(self, lams):
        n = [len(self.step_cols)] + [len(self.glob_cols)] + [t.shape[1] for t in self.fams.values()]
        per = np.concatenate([np.full(k, np.sqrt(LAM0 / lams[f])) for k, f in zip(n, self.fnames)])
        return per[:n[0]], per[n[0]:]
    def fit(self, data, train_task_ids):
        self._predicted_difficulties = {}
        train = list(train_task_ids)
        self.agents, cells = _cells(data, train)
        A = len(self.agents)
        N, G = self._prep(train, fit=True)
        grp = PROC[self.bench].loc[train].to_numpy()
        fold = np.zeros(len(train), int)
        for i, (_, b_) in enumerate(GroupKFold(5).split(train, groups=grp)): fold[b_] = i
        sizes = [len(self.step_cols)] + [len(self.glob_cols)] + [t.shape[1] for t in self.fams.values()]
        def cv(lams):
            sN, sG = self._scales(lams); Ns, Gs = N * sN, G * sG; tot = 0.0
            for fi in range(5):
                trm = fold[cells[:, 1].astype(int)] != fi
                th, w, v, _ = self._fit_params(Ns, Gs, cells[trm], LAM0, 0.0, A)
                te = cells[~trm]; t = te[:, 1].astype(int); a = te[:, 0].astype(int)
                tot += self._ll(th[a] - Ns[t] @ w - Gs[t] @ v, te[:, 2], te[:, 3])
            return tot
        lams = {f: START for f in self.fnames}
        if self.per_family:
            best = cv(lams)
            for f in self.fnames:
                cur = lams[f]
                for g in GRID:
                    if g == cur: continue
                    trial = dict(lams); trial[f] = g; s = cv(trial)
                    if s < best - 1e-9: best, lams = s, trial
        else:
            best = None
            for g in [0.003, 0.03, 0.3, 3.0]:
                s = cv({f: g for f in self.fnames})
                if best is None or s < best: best, lams = s, {f: g for f in self.fnames}
        self.lams = lams
        sN, sG = self._scales(lams); self.sN_, self.sG_ = sN, sG
        self.th, self.w, self.v, self.gam = self._fit_params(N * sN, G * sG, cells, LAM0, 0.0, A)
        self.ai = {a: i for i, a in enumerate(self.agents)}
        self._test = None
        print(f"   {self.name} {self.bench} lams {lams}", flush=True)
    def _prep(self, ids, fit=False):
        N, G = super()._prep(ids, fit)
        if not fit and hasattr(self, "sN_"): return N * self.sN_, G * self.sG_
        return N, G
