"""Check 2: transfer to tau-Knowledge banking. Train on SOPBench + tau2 (and tau2 only), predict banking difficulty."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, os
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric")
os.environ.setdefault("PYTHONHASHSEED", "0")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from ladder import Ladder
import prompts as P
HERE = Path(S + "/ideas7/evo_check"); EVO = Path(S + "/ideas7/evo"); I3 = Path(S + "/ideas3")
R = Path(ZR.REPO)
def tab(p):
    d = pd.read_csv(p); return d.set_index(d.columns[0])
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
B = ["sopbench", "tau2", "tauk_banking"]
TK = {b: pd.read_csv(R / "data" / b / "tasks.csv").set_index("case_id") for b in B}
rub = pd.concat([tab(S + f"/rubric/features/rub_proc_{b}.csv") for b in B])
ad = pd.concat([tab(R / "data" / b / "features" / "adele.csv") for b in B])
plan = pd.concat([tab(I3 / "plan_tools/features/plan3_sopbench.csv"), tab(I3 / "plan_tools/features/plan3_tau2.csv"), tab(HERE / "bank_plan.csv")])
f1 = pd.read_csv(EVO / "feat_f1.csv").set_index("case_id"); f1 = pd.concat([f1, tab(HERE / "bank_f1.csv")])
dom = pd.concat([TK[b].domain for b in B]); f1 = f1.loc[dom.index]
f1c = f1 - f1.groupby(dom).transform("mean"); f1c.columns = ["evo1_" + c for c in f1c.columns]
SOPONLY = ["most_diff_check"]  # the only F1 feature whose input exists only for SOPBench; the others are computed for tau2/banking from dry-run, plans, message, scenario
class Data:
    def __init__(self, bench_list, test):
        self.responses = {}
        for b in bench_list:
            for l in open(R / "data" / b / "responses.jsonl"):
                d = json.loads(l); self.responses[f"{b}|{d['subject_id']}"] = d["responses"]
        self.train_abilities = pd.DataFrame(index=list(self.responses)); self.test_tasks = test
def run(name, bench_list, extra, step=TYPES, glob=GLOB):
    test = list(TK["tauk_banking"].index)
    train = [c for b in bench_list for c in TK[b].index]
    data = Data(bench_list, test)
    m = Ladder(2, step, glob, rub, extra_table=extra, name=name); m.fit(data, train)
    m.predict_probability(data, list(data.responses)[0], test[0])
    return pd.Series({t: m._predicted_difficulties[t] for t in test})
y = pd.read_csv(R / "data/tauk_banking/irt/1d_1pl/items.csv", index_col=0).b.loc[TK["tauk_banking"].index]
base_ex = pd.concat([ad, plan], axis=1)
def with_f1(cols=None): return pd.concat([base_ex, f1c if cols is None else f1c[cols]], axis=1)
preds = {}
for tr_name, bl in [("sop+tau2", ["sopbench", "tau2"]), ("tau2only", ["tau2"])]:
    preds[f"c_lad[{tr_name}]"] = run("c_lad", bl, base_ex)
    preds[f"c_lad+F1[{tr_name}]"] = run("c_lad+F1", bl, with_f1())
    tau_ok = ["evo1_" + c for c in f1.columns if c not in SOPONLY]
    preds[f"c_lad+F1_nomdc[{tr_name}]"] = run("c_lad+F1_nomdc", bl, with_f1(tau_ok))
# F1 features alone: ridge on domain-centred IRT b of train cases
from sklearn.linear_model import RidgeCV
irt = pd.concat([pd.read_csv(R / "data" / b / "irt/1d_1pl/items.csv", index_col=0).b for b in ["sopbench", "tau2"]])
for tr_name, bl in [("sop+tau2", ["sopbench", "tau2"]), ("tau2only", ["tau2"])]:
    ids = [c for b in bl for c in TK[b].index]; yy = irt.loc[ids]; yy = yy - yy.groupby(dom.loc[ids]).transform("mean")
    X = f1c.loc[ids]; mu, sd = X.mean(), X.std() + 1e-9
    r = RidgeCV(alphas=[1, 10, 100, 1000, 10000]).fit((X - mu) / sd, yy)
    preds[f"F1_alone[{tr_name}]"] = pd.Series(r.predict((f1c.loc[y.index] - mu) / sd), index=y.index)
for c in f1.columns: preds[f"raw:{c}"] = f1.loc[y.index, c]
preds["raw:adele_mean"] = ad.loc[y.index].mean(1)
P_ = pd.DataFrame(preds); P_.to_csv(HERE / "bank_preds.csv")
rng = np.random.default_rng(20261008); NB = 2000; n = len(y); idx = rng.integers(0, n, (NB, n))
def rho(a, b): return spearmanr(a, b)[0] if np.std(a) > 0 and np.std(b) > 0 else np.nan
yv = y.values; boots = {c: np.array([rho(P_[c].values[i], yv[i]) for i in idx]) for c in P_}
lines = [f"banking tasks {n}; target = banking IRT difficulty b"]
for c in P_:
    lines.append(f"{c:36s} rho {rho(P_[c].values, yv):+.3f} [{np.nanpercentile(boots[c], 2.5):+.3f},{np.nanpercentile(boots[c], 97.5):+.3f}]")
for tr in ["sop+tau2", "tau2only"]:
    for v in [f"c_lad+F1[{tr}]", f"c_lad+F1_nomdc[{tr}]"]:
        a = f"c_lad[{tr}]"; d = boots[v] - boots[a]
        lines.append(f"paired {v} - {a}: {rho(P_[v].values, yv) - rho(P_[a].values, yv):+.3f} [{np.nanpercentile(d, 2.5):+.3f},{np.nanpercentile(d, 97.5):+.3f}]  P(diff>0)={np.nanmean(d > 0):.2f}")
open(HERE / "check2.txt", "w").write("\n".join(lines)); print("\n".join(lines))
