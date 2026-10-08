"""Extra metrics for the ideas round (item 8 of the list), computed on harness output folders.

All metrics use the same out-of-fold predictions as the protocol evaluator (obs_<scheme>.csv.gz written by
harness.run_all). Predicted task difficulty is the protocol's: minus the mean logit of P(success) over agents.

  procedure-level rho   SOPBench only. Mean predicted and mean fitted difficulty per procedure, Spearman inside
                        each domain. Main version on the new-domain split (one fold per domain, so all procedures of
                        a domain come from one fold model). Paired bootstrap over procedures.
  label-conditional rho SOPBench only. Protocol rho computed inside (fold, domain, allow/refuse) cells, so the hidden
                        case state (should the request be performed or refused) is held fixed. Evaluation only, the
                        label is never a feature.
  hard-case recall      Share of the truly hardest 20% of tasks of a (fold, domain) cell that the method puts in its own
                        hardest 20% (chance = 0.2). The cases a router would hand to a person.
  selective lift        Within one agent: success rate on the half of its test tasks the method rates safest, minus its
                        success rate on all its test tasks (chance = 0). The gain from automating only the safer half.
  ceiling share         rho divided by the rho of an oracle that knows each procedure's mean fitted difficulty
                        (what any procedure-level documentation feature could reach at best), SOPBench only.

  import metrics_extra as X
  X.report(out_dir, ["adele", "my_method"], extra_dirs={"lad2_adele": rubric_dir}, combos=[("my_method", "adele")])
writes out_dir/extra_metrics.json and prints a table.
"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr

REPO = Path(ZR.REPO)
KEY = ["seed", "fold", "agent", "case_id"]
NEW = [("sopbench", "new_procedures"), ("sopbench", "new_domain"), ("tau2", "new_procedures"), ("tau2", "new_domain")]
N_BOOT = 1000


def tasks_of(bench, data_dir=None):
    d = Path(data_dir or REPO / "data") / bench
    t = pd.read_csv(d / "tasks.csv")
    t["b"] = t.case_id.map(pd.read_csv(d / "irt/1d_1pl/items.csv", index_col=0).b)
    return t


def load(out_dir, bench, scheme, cols, extra_dirs=None, combos=()):
    m = pd.read_csv(Path(out_dir) / bench / f"obs_{scheme}.csv.gz")
    for col, d in (extra_dirs or {}).items():
        p = Path(d) / bench / f"obs_{scheme}.csv.gz"
        if col in m or not p.exists():
            continue
        e = pd.read_csv(p, usecols=lambda c: c in KEY + [col]).drop_duplicates(KEY)
        if col in e:
            m = m.merge(e, on=KEY, how="left")
    c = lambda x: np.clip(x, 1e-4, 1 - 1e-4)  # noqa: E731
    for combo in combos:
        if all(k in m for k in combo):
            m["+".join(combo)] = expit(np.mean([logit(c(m[k])) for k in combo], axis=0))
    keep = [x for x in cols + ["+".join(cb) for cb in combos] if x in m and m[x].notna().all()]
    return m, keep


def difficulty(m, col):
    u = m.drop_duplicates(KEY)
    return (u.assign(l=-logit(np.clip(u[col], 1e-4, 1 - 1e-4))).groupby(["seed", "fold", "case_id"]).l.mean()
            .reset_index())


def _rho_cells(df, cells, a="l", b="b"):
    num = den = 0.0
    for _, g in df.groupby(cells):
        n = len(g)
        if n < 3:
            continue
        pr = n * (n - 1) / 2
        if g[a].nunique() > 1 and g[b].nunique() > 1:
            num += spearmanr(g[a], g[b]).correlation * pr
        den += pr
    return num / den if den else np.nan


def _hard_recall(df, cells, q=0.2):
    hit = tot = 0.0
    for _, g in df.groupby(cells):
        n = len(g)
        k = int(round(q * n))
        if n < 5 or k < 1:
            continue
        true_hard = set(g.nlargest(k, "b").case_id)
        pred_hard = set(g.nlargest(k, "l").case_id)
        hit += len(true_hard & pred_hard)
        tot += k
    return hit / tot if tot else np.nan


def _selective_lift(m, col, cover=0.5):
    out, w = [], []
    u = m.groupby(KEY).agg(y=("y", "mean"), p=(col, "first")).reset_index()
    for _, g in u.groupby(["seed", "fold", "agent"]):
        n = len(g)
        k = int(round(cover * n))
        if n < 4 or k < 1 or g.p.nunique() < 2:
            continue
        top = g.sort_values("p", ascending=False, kind="stable").head(k)
        out.append(top.y.mean() - g.y.mean())
        w.append(n)
    return float(np.average(out, weights=w)) if out else np.nan


def _proc_rho(x):
    pm = x.groupby(["domain", "procedure_id"])[["l", "b"]].mean().reset_index()
    return _rho_cells(pm, ["domain"])


def report(out_dir, methods, extra_dirs=None, combos=(), baseline="adele", data_dir=None, write=True):
    out_dir = Path(out_dir)
    res, boot = {}, {}
    for bench, scheme in NEW:
        if not (out_dir / bench / f"obs_{scheme}.csv.gz").exists():
            continue
        t = tasks_of(bench, data_dir)
        m, cols = load(out_dir, bench, scheme, list(dict.fromkeys([baseline] + list(methods) + list(extra_dirs or {}))), extra_dirs, combos)
        key = f"{bench}|{scheme}"
        res[key] = {}
        cols_t = ["case_id", "domain", "procedure_id", "b"] + (["should_succeed"] if "should_succeed" in t else [])
        diffs = {c: difficulty(m, c).merge(t[cols_t], on="case_id") for c in cols}
        for c in cols:
            d = diffs[c]
            r = {"hard_recall20": float(np.mean([_hard_recall(g, ["fold", "domain"]) for _, g in d.groupby("seed")])),
                 "selective_lift50": _selective_lift(m, c)}
            if bench == "sopbench":
                r["rho_given_label"] = float(np.mean([_rho_cells(g, ["fold", "domain", "should_succeed"])
                                                      for _, g in d.groupby("seed")]))
                if scheme == "new_domain":
                    r["rho_procedure_level"] = float(_proc_rho(d[d.seed == 0]))
            res[key][c] = r
        if bench == "sopbench" and scheme == "new_domain":
            # oracle ceilings on the same cells, and the paired bootstrap of the procedure-level rho
            g = diffs[cols[0]][diffs[cols[0]].seed == 0].copy()
            g["l"] = g.groupby("procedure_id").b.transform("mean")
            res[key]["_oracle_procedure_mean"] = {"rho_protocol": float(_rho_cells(g, ["fold", "domain"])),
                                                  "rho_procedure_level": 1.0}
            g["l"] = g.groupby(["procedure_id", "should_succeed"]).b.transform("mean")
            res[key]["_oracle_procedure_x_label"] = {"rho_protocol": float(_rho_cells(g, ["fold", "domain"]))}
            rng = np.random.default_rng(20261004)
            pm = {c: diffs[c][diffs[c].seed == 0].groupby(["domain", "procedure_id"])[["l", "b"]].mean().reset_index()
                  for c in cols}
            procs = pm[cols[0]][["domain", "procedure_id"]].values.tolist()
            by_dom = {}
            for dmn, p in procs:
                by_dom.setdefault(dmn, []).append(p)
            draws = {c: [] for c in cols}
            for _ in range(N_BOOT):
                pick = {dmn: rng.choice(ps, len(ps)) for dmn, ps in by_dom.items()}
                for c in cols:
                    num = den = 0.0
                    idx = pm[c].set_index("procedure_id")
                    for dmn, ps in pick.items():
                        sub = idx.loc[ps]
                        n = len(sub)
                        if n < 3 or sub.l.nunique() < 2 or sub.b.nunique() < 2:
                            continue
                        pr = n * (n - 1) / 2
                        num += np.corrcoef(rankdata(sub.l), rankdata(sub.b))[0, 1] * pr
                        den += pr
                    draws[c].append(num / den if den else np.nan)
            for c in cols:
                res[key][c]["rho_procedure_level_ci"] = [float(np.nanpercentile(draws[c], 2.5)),
                                                         float(np.nanpercentile(draws[c], 97.5))]
                if c != baseline:
                    dd = np.array(draws[c]) - np.array(draws[baseline])
                    res[key][c]["rho_procedure_level_minus_baseline_ci"] = [float(np.nanpercentile(dd, 2.5)),
                                                                            float(np.nanpercentile(dd, 97.5))]
    # summaries over the four scenarios
    allc = sorted({c for k in res for c in res[k] if not c.startswith("_")})
    summ = {}
    for c in allc:
        hv = [res[k][c]["hard_recall20"] for k in res if c in res[k]]
        sv = [res[k][c]["selective_lift50"] for k in res if c in res[k]]
        summ[c] = {"hard_recall20_mean": float(np.nanmean(hv)), "selective_lift50_mean": float(np.nanmean(sv))}
        sd = res.get("sopbench|new_domain", {}).get(c, {})
        if "rho_procedure_level" in sd:
            summ[c].update({k: sd[k] for k in sd if k.startswith("rho_procedure_level")})
    res["summary"] = summ
    if write:
        (out_dir / "extra_metrics.json").write_text(json.dumps(res, indent=1))
    print(f"{'method':34s} {'proc-rho SOP new dom':>22s} {'vs base CI':>18s} {'hard@20':>8s} {'lift50':>7s}")
    for c in allc:
        s = summ[c]
        pr = s.get("rho_procedure_level", np.nan)
        ci = s.get("rho_procedure_level_minus_baseline_ci", [np.nan, np.nan])
        print(f"{c:34s} {pr:22.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] {s['hard_recall20_mean']:8.3f} {s['selective_lift50_mean']:7.3f}")
    return res
