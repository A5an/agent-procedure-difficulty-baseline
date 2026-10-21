"""usage: run.py pre | main. tau2 only, seed 0 (SEEDS=1). See PREREG.md."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, importlib.util
S = ZR.SCREEN
mode = sys.argv[1]
sys.argv = [sys.argv[0], "main"]
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric"); sys.path.insert(0, S + "/ideas3/conditions")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
spec = importlib.util.spec_from_file_location("combo_run", S + "/ideas3/combo/run.py"); C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
import harness as H
from cpulock import cpu_lock
import fs
from sklearn.model_selection import GroupKFold
HERE = Path(__file__).parent
tasks = pd.read_csv(fs.REPO / "data/tau2/tasks.csv"); IDS = tasks.case_id.tolist()
def folds(scheme, seed=0):
    if scheme == "new_procedures":
        it = GroupKFold(5, shuffle=True, random_state=seed).split(IDS, groups=tasks.procedure_id)
    else:
        doms = sorted(tasks.domain.unique()); it = [((tasks.domain != d).to_numpy().nonzero()[0], (tasks.domain == d).to_numpy().nonzero()[0]) for d in doms]
    return [([IDS[i] for i in a], [IDS[i] for i in b]) for a, b in it]
def scheme_of(tr, te):
    return "new_domain" if not ({fs.DOM[c] for c in tr} & {fs.DOM[c] for c in te}) else "new_procedures"
_FC = {}
def fold_feats(tr, te):
    key = (hash(tuple(sorted(tr))), hash(tuple(sorted(te))))
    if key not in _FC:
        A = fs.stage_a(tr, te)
        B, nf = (None, 0) if os.environ.get('NOB') else fs.b_table(tr, te, scheme_of(tr, te))
        _FC[key] = (A, B, nf)
    return _FC[key]

class GF(C.Grouped):
    def fit(self, data, tr):
        A, B, _ = fold_feats(list(tr), list(data.test_tasks))
        self.tabs = dict(self.tabs); self.tabs["A"] = A
        if B is not None: self.tabs["B"] = B
        super().fit(data, tr)
class LadF(C.Ladder):
    def __init__(self, fam, base, rub, name):
        super().__init__(2, C.TYPES, C.GLOB, rub, extra_table=None, name=name); self.fam, self.base = fam, base
    def fit(self, data, tr):
        A, B, _ = fold_feats(list(tr), list(data.test_tasks))
        parts = [self.base] + ([{"A": A, "B": B}[f] for f in self.fam])
        ids = list(tr) + list(data.test_tasks)
        self.extra = pd.concat([p.loc[ids] for p in parts], axis=1); self._test = None
        super().fit(data, tr)

def make_preds(src, M, bench):
    t = C.static_tables(bench)
    ad = t["ADELE"]; ad_plan = pd.concat([t["ADELE"], t["PLAN"]], axis=1); ad_only = t["ADELE"]
    if os.environ.get("NOB"):
        return {"A": GF("A", bench, ["A"]), "A_adele": GF("A_adele", bench, ["ADELE", "A"]),
                "c_lad": C.Ladder(2, C.TYPES, C.GLOB, t["RUB"], extra_table=ad_plan, name="c_lad"),
                "c_lad_A": LadF(["A"], ad_plan, t["RUB"], "c_lad_A")}
    return {"A": GF("A", bench, ["A"]), "A_adele": GF("A_adele", bench, ["ADELE", "A"]), "B": GF("B", bench, ["B"]),
            "B_adele": GF("B_adele", bench, ["ADELE", "B"]),
            "c_lad": C.Ladder(2, C.TYPES, C.GLOB, t["RUB"], extra_table=ad_plan, name="c_lad"),
            "c_lad_A": LadF(["A"], ad_plan, t["RUB"], "c_lad_A"), "c_lad_B": LadF(["B"], ad_plan, t["RUB"], "c_lad_B"),
            "c_lad_Brep": LadF(["B"], ad_only, t["RUB"], "c_lad_Brep")}

if mode == "pre":
    rows = []; nf = {}
    for scheme in ["new_procedures", "new_domain"]:
        for k, (tr, te) in enumerate(folds(scheme)):
            A, B, n = fold_feats(tr, te); nf[f"{scheme}{k}"] = n
            for c in te:
                g = fs.gold_stats(fs._GOLD[c], fs.DOM[c])  # validation only, test gold never feeds features
                rows.append(dict(scheme=scheme, case=c, dom=fs.DOM[c], A_len=A.loc[c, "A_len"], A_writes=A.loc[c, "A_writes"], A_alllen=A.loc[c, "A_alllen"],
                                 B_calls=B.loc[c, "plan_calls"], B_writes=B.loc[c, "plan_writes"], Z_calls=fs.PLAN3.loc[c, "plan_calls"], Z_writes=fs.PLAN3.loc[c, "plan_writes"],
                                 g_len=g[0], g_writes=g[1], g_all=g[2]))
            print(scheme, k, "calls so far", fs.NCALLS[0], flush=True)
    d = pd.DataFrame(rows); d.to_csv(HERE / "validation_rows.csv", index=False)
    def r(a, b):
        return None if d_[a].nunique() < 2 or d_[b].nunique() < 2 else round(float(spearmanr(d_[a], d_[b])[0]), 3)
    res = {}
    for sch, d0 in d.groupby("scheme"):
        for dom, d_ in list(d0.groupby("dom")) + [("all", d0)]:
            res[f"{sch}|{dom}"] = dict(n=len(d_), zero_shot_calls_vs_len=r("Z_calls", "g_len"), B_calls_vs_len=r("B_calls", "g_len"), A_len_vs_len=r("A_len", "g_len"),
                zero_shot_writes_vs_writes=r("Z_writes", "g_writes"), B_writes_vs_writes=r("B_writes", "g_writes"), A_writes_vs_writes=r("A_writes", "g_writes"),
                zero_calls_vs_all=r("Z_calls", "g_all"), B_calls_vs_all=r("B_calls", "g_all"), A_all_vs_all=r("A_alllen", "g_all"),
                mean_gold_len=round(float(d_.g_len.mean()), 2), mean_zero=round(float(d_.Z_calls.mean()), 2), mean_B=round(float(d_.B_calls.mean()), 2), mean_A=round(float(d_.A_len.mean()), 2))
    res["parse_fail_per_fold"] = nf; res["llm_calls_this_process"] = fs.NCALLS[0]
    json.dump(res, open(HERE / "validation.json", "w"), indent=1); print(json.dumps(res, indent=1))
else:
    out = HERE / ("out_A" if os.environ.get("NOB") else "out_main")
    with cpu_lock("fewshot_plan"):
        H.run_all(out, make_preds, benches=["tau2"], schemes=["new_procedures", "new_domain"])
    print("run done")
