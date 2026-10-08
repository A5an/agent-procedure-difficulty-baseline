"""Score existing out-of-fold predictions (no refit) against tiered targets. usage: score.py [refit]"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.special import logit
from scipy.stats import rankdata
S = ZR.SCREEN
sys.path.insert(0, S + "/ideas")
from cpulock import cpu_lock
import tiers as TI
HERE = Path(__file__).parent
R = str(TI.REPO) + "/results"
KEY = ["seed", "fold", "agent", "case_id"]
NB = 300
SRC = {  # method column -> dir holding obs files
    "adele": R, "length": R, "human_time": R, "oracle": R,
    "lad2_adele": S + "/rubric/out/run", "grp_proc_adele": S + "/rubric/out/run",
    "dry_cat_adele": S + "/ideas/dryrun/out/run", "dry_grp_adele": S + "/ideas/dryrun/out/run",
    "lupi": S + "/ideas/lupi/out",
}
NEW = [("sopbench", "new_procedures"), ("sopbench", "new_domain"), ("tau2", "new_procedures"), ("tau2", "new_domain")]
c = lambda x: np.clip(x, 1e-4, 1 - 1e-4)

def case_diff(bench, scheme, srcs):
    out = None
    for col, d in srcs.items():
        p = Path(d) / bench / f"obs_{scheme}.csv.gz"
        if not p.exists(): continue
        hdr = pd.read_csv(p, nrows=1).columns
        if col not in hdr: continue
        m = pd.read_csv(p, usecols=KEY + [col]).drop_duplicates(KEY)
        m["l"] = -logit(c(m[col]))
        g = m.groupby(["seed", "fold", "case_id"]).l.mean().rename(col).reset_index()
        out = g if out is None else out.merge(g, on=["seed", "fold", "case_id"], how="left")
    return out

def targets(bench):
    T, Y, acc = TI.tiers(bench)
    t = pd.read_csv(f"{TI.REPO}/data/{bench}/tasks.csv").set_index("case_id")
    out = pd.DataFrame(index=t.index)
    out["all_irt"] = pd.read_csv(f"{TI.REPO}/data/{bench}/irt/1d_1pl/items.csv", index_col=0).b
    for k in ["all", "all_clean", "top5", "top8", "bot5", "bot8"]:
        out["fail_" + k] = 1 - Y[T[k]].mean(axis=1)
    if "should_succeed" in t: out["label"] = t.should_succeed.astype(float)
    return t, out

def rho_stats(F, cell_cols, M, Tg, unit, counts, seeds_col="seed"):
    """pooled within-cell Spearman, averaged over seeds. rows replicated by counts[unit]. returns array [len(M), len(Tg)]."""
    res = []
    for _, g in F.groupby(seeds_col):
        num = np.zeros((len(M), len(Tg))); den = 0.0
        for _, cg in g.groupby(cell_cols):
            w = counts[cg[unit].values]
            idx = np.repeat(np.arange(len(cg)), w)
            n = len(idx)
            if n < 3: continue
            pr = n * (n - 1) / 2; den += pr
            X = cg[M].values[idx].T; Y = cg[Tg].values[idx].T
            rx = rankdata(X, axis=1); ry = rankdata(Y, axis=1)
            rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
            sx = np.sqrt((rx ** 2).sum(1)); sy = np.sqrt((ry ** 2).sum(1))
            with np.errstate(all="ignore"):
                r = (rx @ ry.T) / np.outer(sx, sy)
            num += pr * np.nan_to_num(r, nan=0.0)
        res.append(num / den if den else np.full(num.shape, np.nan))
    return np.mean(res, axis=0)

def boot(F, cell_cols, M, Tg, unit_codes_by_domain, n_units, rng, nb=NB):
    """bootstrap over procedures stratified by domain. unit_codes_by_domain: list of arrays of unit codes."""
    pt = rho_stats(F, cell_cols, M, Tg, "u", np.ones(n_units, int))
    draws = []
    for _ in range(nb):
        cnt = np.zeros(n_units, int)
        for codes in unit_codes_by_domain:
            np.add.at(cnt, rng.choice(codes, len(codes)), 1)
        draws.append(rho_stats(F, cell_cols, M, Tg, "u", cnt))
    return pt, np.array(draws)

def build(srcs, tgt_cols, benches=("sopbench", "tau2")):
    """returns dict[(bench, scheme)] -> wide frame, plus task tables"""
    out = {}
    for b in benches:
        t, tg = targets(b)
        for sch in ["new_procedures", "new_domain"]:
            D = case_diff(b, sch, srcs)
            if D is None: continue
            D = D.merge(t[["domain", "procedure_id"]].reset_index(), on="case_id").merge(tg.reset_index(), on="case_id")
            out[(b, sch)] = D
    return out

def main(mode):
    rng = np.random.default_rng(20261005)
    if mode == "refit":
        tiers_r = ["top5", "top8", "bot5"]
        srcs_by = {t: {"adele": f"{HERE}/refit/{t}", "lad2_adele": f"{HERE}/refit/{t}"} for t in tiers_r}
    TG = ["all_irt", "fail_all_clean", "fail_top5", "fail_top8", "fail_bot5", "fail_bot8"]
    results = {}
    if mode == "existing":
        sets = {"existing": SRC}
    else:
        sets = {f"refit_{t}": srcs_by[t] for t in tiers_r}
    for name, srcs in sets.items():
        W = build(srcs, TG)
        for bench in ["sopbench", "tau2"]:
            Ds = {s: W[(bench, s)] for s in ["new_procedures", "new_domain"] if (bench, s) in W}
            if not Ds: continue
            M0 = [m for m in srcs if all(m in D for D in Ds.values())]
            # precheck (SOP only, L2)
            extra = []
            if bench == "sopbench" and mode == "existing":
                for var in ["A", "B"]:
                    res = pd.DataFrame(json.load(open(f"{S}/ideas2/compile/results_{var}.json")))
                    res["perf_hat"] = [int(p in tl) for p, tl in zip(res.procedure, res.tools)]
                    ph = res.set_index("case_id").perf_hat
                    for D in Ds.values():
                        D[f"pre{var}_only"] = D.case_id.map(ph) + 1e-6 * D["adele"]
                        D[f"pre{var}_base"] = 2.26 * D.case_id.map(ph) + D["adele"]
                    extra += [f"pre{var}_only", f"pre{var}_base"]
                for D in Ds.values():
                    D["label_oracle"] = D["label"] + 1e-6 * D["adele"]
                extra.append("label_oracle")
            M = M0 + extra
            for level in ["case", "proc"]:
                for sch, D in Ds.items():
                    D = D.copy()
                    procs = sorted(D.procedure_id.unique()); code = {p: i for i, p in enumerate(procs)}
                    D["u"] = D.procedure_id.map(code)
                    by_dom = [np.array(sorted(g.u.unique())) for _, g in D.groupby("domain")]
                    if level == "proc":
                        cols = M + TG
                        P = D.groupby(["seed", "domain", "procedure_id"], as_index=False)[cols].mean()
                        P["u"] = P.procedure_id.map(code)
                        F, cells = P, ["domain"]
                    else:
                        F, cells = D, ["fold", "domain"]
                    pt, dr = boot(F, cells, M, TG, by_dom, len(procs), np.random.default_rng(rng.integers(1 << 31)))
                    results[(name, bench, sch, level)] = (M, TG, pt, dr)
                    print(name, bench, sch, level, "done", flush=True)
    return results

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "existing"
    with cpu_lock("tiers_score_" + mode):
        res = main(mode)
    import pickle
    pickle.dump(res, open(HERE / f"score_{mode}.pkl", "wb"))
