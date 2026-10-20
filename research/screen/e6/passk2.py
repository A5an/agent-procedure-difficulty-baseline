"""E6, second pass: pass^k on tau2 from the out-of-fold P of the baseline (adele) and of the best model (f2_all_fc),
raw and recalibrated (one logit shift fitted on the other folds), each with independent attempts and beta-binomial.
The first pass (../targets/passk.py) used the baseline P only and left recalibration open."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import comb, expit, logit
sys.path.insert(0, ZR.SCREEN + "/targets")
from passk import fit_phi, passk_bb

OBS = ZR.SCREEN + "/ideas10/fc/out/tau2/obs_{}.csv.gz"
KS = [1, 2, 3, 4]
COLS = ["adele", "f2_all_fc"]


def fit_shift(s, n, p):
    lp = logit(np.clip(p, 1e-4, 1 - 1e-4))
    f = lambda c: -(s * np.log(expit(lp + c) + 1e-12) + (n - s) * np.log(1 - expit(lp + c) + 1e-12)).sum()
    return float(minimize_scalar(f, bounds=(-5, 5), method="bounded").x)


def table(scheme):
    o = pd.read_csv(OBS.format(scheme), usecols=["seed", "fold", "agent", "case_id", "y"] + COLS)
    agg = {"s": ("y", "sum"), "n": ("y", "size")}
    agg.update({c: (c, "first") for c in COLS})
    return o.groupby(["seed", "fold", "agent", "case_id"]).agg(**agg).reset_index()


def run(scheme):
    g = table(scheme)
    parts = []
    for (seed, fold), te in g.groupby(["seed", "fold"]):
        tr = g[(g.seed == seed) & (g.fold != fold)]
        te = te.copy()
        for c in COLS:
            sh = fit_shift(tr.s.values, tr.n.values, tr[c].values)
            ptr_rec = expit(logit(tr[c].clip(1e-4, 1 - 1e-4)) + sh)
            te[c + "|raw"] = te[c].clip(1e-4, 1 - 1e-4)
            te[c + "|rec"] = expit(logit(te[c].clip(1e-4, 1 - 1e-4)) + sh)
            te[c + "|raw|phi"] = fit_phi(tr.s.values, tr.n.values, tr[c].clip(1e-4, 1 - 1e-4).values)
            te[c + "|rec|phi"] = fit_phi(tr.s.values, tr.n.values, ptr_rec.values)
            te[c + "|shift"] = sh
        parts.append(te)
    d = pd.concat(parts)
    res = {"cells_per_seed": int(len(d) / d.seed.nunique())}
    for c in COLS:
        for v in ("raw", "rec"):
            res[f"{c}|{v}|phi_mean"] = float(d[f"{c}|{v}|phi"].mean())
        res[f"{c}|shift_mean"] = float(d[f"{c}|shift"].mean())
    briers = {}
    for k in KS:
        m = d[d.n >= k]
        frac = comb(m.s.values, k) / comb(m.n.values, k)
        res[f"k{k}_obs"] = float(frac.mean())
        for c in COLS:
            for v in ("raw", "rec"):
                p = m[f"{c}|{v}"].values
                for sch, pk in (("indep", p ** k), ("bb", passk_bb(p, m[f"{c}|{v}|phi"].values, k))):
                    if k == 1 and sch == "bb": continue
                    b = pk ** 2 - 2 * pk * frac + frac
                    key = f"{c}|{v}|{sch}"
                    res[f"k{k}|{key}"] = {"brier": float(b.mean()), "mean_pred": float(pk.mean())}
                    briers[(k, key)] = pd.DataFrame({"t": m.case_id.values, "b": b})
    # paired bootstrap over tasks: best-model recalibrated beta-binomial vs the plan's default (baseline, raw, independent)
    rng = np.random.default_rng(0)
    for k in KS[1:]:
        for a, b in ((f"f2_all_fc|rec|bb", "adele|raw|indep"), ("f2_all_fc|rec|bb", "f2_all_fc|rec|indep"), ("f2_all_fc|rec|indep", "adele|raw|indep")):
            x = briers[(k, a)].assign(b=briers[(k, a)].b.values - briers[(k, b)].b.values).groupby("t").b.agg(["sum", "count"])
            sums, cnts = x["sum"].values, x["count"].values
            bs = [sums[ix].sum() / cnts[ix].sum() for ix in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
            res[f"k{k}|{a} - {b}"] = {"diff": float(sums.sum() / cnts.sum()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return res


if __name__ == "__main__":
    out = {s: run(s) for s in ("new_procedures", "new_domain")}
    json.dump(out, open(ZR.SCREEN + "/e6/passk2.json", "w"), indent=1)
    for s, r in out.items():
        print("==", s, "cells/seed", r["cells_per_seed"])
        print("   shifts", {c: round(r[f"{c}|shift_mean"], 3) for c in COLS}, "phi", {k: round(v, 2) for k, v in r.items() if k.endswith("phi_mean")})
        for k in KS:
            row = [f"k={k} obs {r[f'k{k}_obs']:.3f}"]
            for key in [x for x in r if x.startswith(f"k{k}|") and " - " not in x]:
                row.append(f"{key[len(f'k{k}|'):]} {r[key]['brier']:.4f}/{r[key]['mean_pred']:.3f}")
            print("  ", " | ".join(row))
        for key in [x for x in r if " - " in x]:
            print("  ", key, round(r[key]["diff"], 4), [round(v, 4) for v in r[key]["ci"]])
