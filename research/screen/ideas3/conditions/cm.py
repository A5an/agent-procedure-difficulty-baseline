"""Condition model and case predictors (L0 and L2 tracks).  See REPORT in the final message.
Variants fixed in advance:
  l0_alone  : P_case = pp * prod_c(1 - skip_perf_c) + (1 - pp) * mean_c(1 - skip_fail_c), pp = 1 - rubric p_refusal; agent offset
  l0_adele  : ridge of ADeLe (18) + z (z = out-of-fold logit of the l0 case score) on training betas, P = sigmoid(theta - beta)
  l0_agent  : like l0_alone, skip rates per agent family (residual ridge on the global model)
  l0_prior  : like l0_alone but pp = training share of must-perform cases (documentation-free perform prior)
  l2_alone  : compiled program decision (action called or not). perform -> prod(1 - skip_perf); refuse -> mean over the candidate
              conditions (those that use the last tool the program called) of (1 - skip_fail)
  l2_adele, l2_agent as above
  l2_truth  : DIAGNOSTIC ONLY (privileged): true label and true failing conditions from the rule tree
"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from targets import family, FAMS, FLAG
S = ZR.SCREEN
REPO = ZR.REPO
HERE = ZR.SCREEN + "/ideas3/conditions"
ALPHAS = [10, 100, 1000, 3000, 10000, 30000]
FAM_ALPHA = 300.0

cd = pd.read_csv(f"{HERE}/conds.csv", keep_default_na=False)
FE = pd.read_csv(f"{HERE}/cond_feats.csv")
FEATS = [c for c in FE.columns if c not in ("unit", "idx")]
cd = cd.merge(FE.drop(columns=[c for c in FE.columns if c in cd.columns and c not in ("unit", "idx")]), on=["unit", "idx"])
cd["key"] = list(zip(cd.unit, cd.idx))
OUT = pd.read_csv(f"{HERE}/cond_outcomes.csv.gz")[["case_id", "unit", "idx", "fam", "should_succeed", "blocking", "skip"]]
OUT["procedure_id"] = OUT.case_id.str.rsplit("/", n=1).str[0]
tasks = pd.read_csv(f"{REPO}/data/sopbench/tasks.csv")
UNITS = json.load(open(f"{HERE}/units.json"))
C2U = {c: u for u, cs in UNITS.items() for c in cs}
ROW = {k: i for i, k in enumerate(cd.key)}
UNIT_ROWS = {u: g.index.values for u, g in cd.groupby("unit")}
UNIT_PROC = {u: u.rsplit("/", 1)[0] for u in UNITS}
RUB = pd.read_csv(f"{S}/rubric/features/rub_proc_sopbench.csv").set_index("case_id").p_refusal
SHOULD = tasks.set_index("case_id").should_succeed
_R = pd.DataFrame(json.load(open(f"{S}/ideas2/compile/results_A.json"))).set_index("case_id")
PERF_HAT = pd.Series({c: int(p in tl) for c, p, tl in zip(_R.index, _R.procedure, _R.tools)})
LAST_TOOL = {c: (tl[-1] if tl else None) for c, tl in zip(_R.index, _R.tools)}
ST = pd.read_csv(f"{S}/ideas2/steps/steps.csv", keep_default_na=False)
# candidate conditions per (unit, tool)
CTOOLS = {}
for _, r in cd.iterrows():
    CTOOLS[(r.unit, r.idx)] = set(t for t in r.step_tools.split("|") if t)
for u in cd.unit.unique():
    s = ST[ST.case_id == UNITS[u][0]]
    for _, g in cd[(cd.unit == u) & (cd.step_type == "or_group")].iterrows():
        ch = s[(s.group == g.step_id) & (s.in_or == 1)]
        CTOOLS[(u, g.idx)] = set(t for x in ch.tools for t in x.split("|") if t)
CANDS = {}
for c in _R.index:
    u = C2U[c]; lt = LAST_TOOL[c]
    n = int((cd.unit == u).sum())
    cands = [i for i in range(n) if lt is not None and lt in CTOOLS.get((u, i), set())]
    CANDS[c] = cands
# true failing conditions (privileged, diagnostic only)
TRUEFAIL = {}
_top = ST[(ST.in_or == 0) & (ST.step_type != "action")]
_cdm = cd.set_index(["unit", "step_id"]).idx
for c, g in _top.groupby("case_id"):
    u = C2U.get(c)
    if u is None: continue
    TRUEFAIL[c] = [int(_cdm.loc[(u, sid)]) for sid, tr in zip(g.step_id, g.truth) if tr == 0 and (u, sid) in _cdm.index]


class CondModel:
    """Skip-rate model from text features, fitted on training cases only."""
    def __init__(self, tr, fam_models=False, const=False):
        tr = set(tr)
        o = OUT[OUT.case_id.isin(tr)]
        self.mu_perf = o[o.should_succeed == 1].skip.mean()
        self.mu_fail = o[o.blocking == 1].skip.mean()
        self.pperf_prior = float(SHOULD.loc[list(tr)].mean())
        self.models = {}
        self.pred = {}
        for name, sub in [("perf", o[o.should_succeed == 1]), ("fail", o[o.blocking == 1])]:
            agg = sub.groupby(["unit", "idx"]).skip.agg(["mean", "size"]).reset_index()
            agg["row"] = [ROW[(u, i)] for u, i in zip(agg.unit, agg.idx)]
            X = cd.loc[agg.row, FEATS].values.astype(float)
            m, s = X.mean(0), X.std(0) + 1e-9
            Xs = (X - m) / s
            w = agg["size"].values.astype(float)
            y = agg["mean"].values
            grp = agg.unit.map(UNIT_PROC).values
            best, ba = 1e9, 100
            if len(set(grp)) >= 5:
                for a in ALPHAS:
                    err = 0
                    for a_, b_ in GroupKFold(5).split(Xs, y, grp):
                        r = Ridge(alpha=a).fit(Xs[a_], y[a_], sample_weight=w[a_])
                        err += np.sum(w[b_] * (r.predict(Xs[b_]) - y[b_]) ** 2)
                    if err < best: best, ba = err, a
            r = Ridge(alpha=ba).fit(Xs, y, sample_weight=w)
            self.models[name] = (r, m, s, agg, ba)
            Xall = (cd[FEATS].values.astype(float) - m) / s
            self.pred[name] = np.clip(r.predict(Xall), 0.02, 0.98)
            if const: self.pred[name] = np.full(len(cd), self.mu_perf if name == 'perf' else self.mu_fail)
        self.fam = {}
        if fam_models:
            Xall = None
            for name, sub, base in [("perf", o[o.should_succeed == 1], self.pred["perf"]), ("fail", o[o.blocking == 1], self.pred["fail"])]:
                r0, m, s, _, _ = self.models[name]
                Xall = (cd[FEATS].values.astype(float) - m) / s
                for f in FAMS:
                    sf = sub[sub.fam == f].groupby(["unit", "idx"]).skip.agg(["mean", "size"]).reset_index()
                    if len(sf) < 10:
                        self.fam[(name, f)] = base; continue
                    rows = np.array([ROW[(u, i)] for u, i in zip(sf.unit, sf.idx)])
                    res = sf["mean"].values - base[rows]
                    rr = Ridge(alpha=FAM_ALPHA).fit(Xall[rows], res, sample_weight=sf["size"].values.astype(float))
                    self.fam[(name, f)] = np.clip(base + rr.predict(Xall), 0.02, 0.98)

    def sk(self, name, fam=None):
        return self.pred[name] if fam is None or (name, fam) not in self.fam else self.fam[(name, fam)]

    def case_score(self, c, track, fam=None, pp_mode="rubric", truth=False):
        u = C2U[c]
        rows = UNIT_ROWS.get(u, np.array([], dtype=int))
        sp, sf = self.sk("perf", fam), self.sk("fail", fam)
        if len(rows) == 0:
            Pp, Pr = 1.0, 1 - self.mu_fail
            Pr = 1 - self.mu_fail
            pf = lambda idx: Pr
        else:
            Pp = float(np.prod(1 - sp[rows]))
        refuse_p = lambda idx: float(np.mean(1 - sf[rows[idx]])) if len(idx) else float(np.mean(1 - sf[rows]))
        if len(rows) == 0:
            refuse_p = lambda idx: 1 - self.mu_fail
        if track == "plain":
            return 0.3 if PERF_HAT[c] == 1 else 0.7
        if truth:
            if SHOULD[c] == 1: return Pp
            return refuse_p(TRUEFAIL.get(c, []))
        if track == "l2":
            if PERF_HAT[c] == 1: return Pp
            return refuse_p(CANDS[c])
        pp = self.pperf_prior if pp_mode == "prior" else 1 - float(RUB[c])
        return pp * Pp + (1 - pp) * refuse_p(list(range(len(rows))))


def offset(y_logit_fn, ys, z):
    """agent intercept a: mean_t sigmoid(a + z_t) = mean_t y_t"""
    target = float(np.mean(ys))
    lo, hi = -10, 10
    for _ in range(50):
        mid = (lo + hi) / 2
        if expit(mid + z).mean() < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2


class Pred:
    def __init__(self, name, track, mode="alone", pp_mode="rubric", truth=False, const=False):
        self.name, self.track, self.mode, self.pp_mode, self.truth, self.const = name, track, mode, pp_mode, truth, const
        self._predicted_difficulties = {}

    def _z(self, cm, cases, fam=None):
        p = np.array([cm.case_score(c, self.track, fam, self.pp_mode, self.truth) for c in cases])
        return logit(np.clip(p, 1e-3, 1 - 1e-3))

    def fit(self, data, tr):
        self._predicted_difficulties = {}
        self.tr = list(tr)
        self.cache = {}
        cm = self.cm = CondModel(tr, fam_models=(self.mode == "agent"), const=self.const)
        agents = list(data.train_abilities.index)
        self.off = {}
        z_tr = {None: self._z(cm, tr)}
        if self.mode == "agent":
            for f in FAMS: z_tr[f] = self._z(cm, tr, f)
        for a in agents:
            f = family(a) if self.mode == "agent" and a not in FLAG else None
            idx = [i for i, t in enumerate(tr) if t in data.responses.get(a, {})]
            ys = [self._y(data, a, tr[i]) for i in idx]
            self.off[a] = offset(None, ys, z_tr[f][idx])
        if self.mode == "adele":
            import harness as H
            ad = pd.read_csv(f"{REPO}/data/sopbench/features/adele.csv", index_col=0)
            self.ad = ad
            # out-of-fold z on training tasks (inner grouped folds by procedure)
            tp = np.array([t.rsplit("/", 1)[0] for t in tr])
            zo = np.zeros(len(tr))
            for a_, b_ in GroupKFold(5).split(tr, groups=tp):
                cmi = CondModel([tr[i] for i in a_])
                zo[b_] = self._z(cmi, [tr[i] for i in b_])
            self.zm, self.zs = zo.mean(), zo.std() + 1e-9
            Xtr = np.column_stack([ad.loc[tr].values, (zo - self.zm) / self.zs])
            self.xm, self.xs = Xtr.mean(0), Xtr.std(0) + 1e-9
            import run_baseline as RB
            self.rm = RidgeCV(alphas=RB.ALPHAS).fit((Xtr - self.xm) / self.xs, data.train_items.loc[tr, "b"].values)

    @staticmethod
    def _y(data, a, t):
        v = data.responses[a][t]
        return float(v["successes"] / v["trials"]) if isinstance(v, dict) else float(v)

    def _zt(self, task, fam):
        k = (task, fam)
        if k not in self.cache:
            self.cache[k] = float(self._z(self.cm, [task], fam)[0])
        return self.cache[k]

    def predict_probability(self, data, agent, task):
        if self.mode == "adele":
            if task not in self._predicted_difficulties:
                te = list(data.test_tasks)
                z = (self._z(self.cm, te) - self.zm) / self.zs
                X = (np.column_stack([self.ad.loc[te].values, z]) - self.xm) / self.xs
                self._predicted_difficulties.update(dict(zip(te, self.rm.predict(X))))
            th = data.train_abilities.loc[agent]
            th = float(th.iloc[0]) if hasattr(th, "iloc") else float(th)
            return float(np.clip(expit(th - self._predicted_difficulties[task]), 1e-4, 1 - 1e-4))
        f = family(agent) if self.mode == "agent" and agent not in FLAG else None
        p = float(np.clip(expit(self.off[agent] + self._zt(task, f)), 1e-4, 1 - 1e-4))
        return p


def make_all(which=None):
    d = make_all_()
    return d if which is None else {k: d[k] for k in which}


def make_all_():
    return {
        "l2_plain": Pred("l2_plain", "plain"), "l0_prior": Pred("l0_prior", "l0", pp_mode="prior"), "l0_const": Pred("l0_const", "l0", const=True), "l2_const": Pred("l2_const", "l2", const=True),
        "l0_alone": Pred("l0_alone", "l0"), "l0_adele": Pred("l0_adele", "l0", "adele"), "l0_agent": Pred("l0_agent", "l0", "agent"),
        "l0_prior": Pred("l0_prior", "l0", pp_mode="prior"),
        "l2_alone": Pred("l2_alone", "l2"), "l2_adele": Pred("l2_adele", "l2", "adele"), "l2_agent": Pred("l2_agent", "l2", "agent"),
        "l2_truth": Pred("l2_truth", "l2", truth=True),
    }
