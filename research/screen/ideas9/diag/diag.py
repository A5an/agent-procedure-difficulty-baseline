"""Round 9 diag: free diagnostic of every case-level feature table. Pure CPU, no LLM."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, glob, warnings, json
sys.path.insert(0, ZR.SCREEN + "/ideas")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from cpulock import cpu_lock
warnings.filterwarnings("ignore")
S = Path(ZR.SCREEN)
D = Path(ZR.REPO + "/data")
OUT = S / "ideas9/diag"
BENCH = ["sopbench", "tau2", "tauk_banking"]

# (relative path with {b}, prefix, family, track, benches or None=all)
SRC = [
 ("REPO:adele.csv", "ad", "ADELE", "L1"),
 ("REPO:length.csv", "len", "LEN", "L0"),
 ("REPO:human_time.csv", "ht", "HUMANTIME", "L1"),
 ("rubric/features/rub_proc_{b}.csv", "rubp", "RUBRIC", "L1"),
 ("rubric/features/rub_lara_{b}.csv", "lara", "RUBRIC_LARA", "L1"),
 ("rubric/features/rub_rpa_{b}.csv", "rpa", "RUBRIC_RPA", "L1"),
 ("rubric/features/rub_len_{b}.csv", "rlen", "RUBRIC", "L1"),
 ("ideas3/plan_tools/features/plan3_{b}.csv", "plan3", "PLAN", "L1"),
 ("ideas3/plan_tools/features/plan3sd_{b}.csv", "plan3sd", "PLAN_SD", "L1"),
 ("ideas3/plan_tau/features/plan_{b}.csv", "plan1", "PLAN", "L1"),
 ("ideas6/planners/features/avg_{b}.csv", "plAvg", "PLAN", "L1"),
 ("ideas6/planners/features/son_{b}.csv", "plSon", "PLAN", "L1"),
 ("ideas6/planners/features/int_{b}.csv", "plInt", "PLAN", "L1"),
 ("ideas6/planners/features/dis_{b}.csv", "plDis", "PLAN_DIS", "L1"),
 ("ideas3/compile_consistency/cons_sopbench.csv", "cons", "CMP", "L1"),
 ("ideas4/code_plan/cmpplus_sopbench.csv", "cmpp", "CMP", "L1"),
 ("ideas4/code_plan/cp_{b}.csv", "cp", "CMP", "L1"),
 ("ideas4/user_scenario/scn_tau2.csv", "scn", "SCN", "L1"),
 ("ideas7/entropy_combo/ent_{b}.csv", "ent", "ENT", "L1-local"),
 ("ideas7/small_model/sm_{b}.csv", "sm", "SMALLMODEL", "L1-local"),
 ("ideas8/textcount/txt_{b}.csv", "txt", "TEXTCOUNT", "L0"),
 ("ideas7/adele_plus/features/A_{b}.csv", "adA", "ADELE_PLUS", "L1"),
 ("ideas7/adele_plus/features/B_{b}.csv", "adB", "ADELE_PLUS", "L1"),
 ("ideas7/adele_plus/features/C_{b}.csv", "adC", "ADELE_PLUS", "L1"),
 ("ideas/dryrun/features/dry_{b}.csv", "dry", "DRYRUN", "L1"),
 ("ideas/fmea/features/fmea_{b}.csv", "fmea", "FMEA", "L1"),
 ("ideas/pairwise/features/bt_{b}.csv", "bt", "PAIRWISE", "L1"),
 ("ideas/pairwise/features/bt_hard_{b}.csv", "bth", "PAIRWISE", "L1"),
 ("ideas3/paper_pilot/features/pf_both_{b}.csv", "pilot", "PAPERPILOT", "L1"),
 ("ideas6/emu_feat/emu_case_features.csv", "emu", "EMULATION", "L1+"),
 ("ideas6/msg_facts/msg_features.csv", "msg", "MSGFACTS", "L1"),
 ("ideas6/personas/features/pers_{b}.csv", "pers", "PERSONAS", "L1+"),
 ("ideas6/synth_label/feats.csv", "synl", "SYNTHLABEL", "L1"),
 ("ideas7/case_signals/fam_act.csv", "act", "CASESIG", "L0"),
 ("ideas7/case_signals/fam_nli.csv", "nli", "CASESIG", "L1-local"),
 ("ideas7/case_signals/fam_ret.csv", "ret", "CASESIG", "L0"),
 ("ideas7/case_signals/fam_bnd.csv", "bnd", "CASESIG", "L0"),
 ("ideas7/tau_graph/features/graph_avg_tau2.csv", "grph", "TAUGRAPH", "L1"),
 ("ideas7/evo_check/bank_plan.csv", "plan3", "PLAN", "L1"),
 ("ideas7/evo_check/bank_dry.csv", "dry", "DRYRUN", "L1"),
]
only = {"cons": ["sopbench"], "cmpp": ["sopbench"], "scn": ["tau2"], "emu": ["sopbench"], "msg": ["sopbench"],
        "synl": ["sopbench"], "act": ["sopbench"], "nli": ["sopbench"], "ret": ["sopbench"], "bnd": ["sopbench"],
        "grph": ["tau2"]}

def load_bench(b):
    t = pd.read_csv(D / b / "tasks.csv").set_index("case_id")
    items = pd.read_csv(D / b / "irt/1d_1pl/items.csv", index_col=0)
    t["b"] = items["b"].reindex(t.index)
    feats, meta, got, skipped = {}, {}, [], []
    for path, pre, fam, trk in SRC:
        if pre in only and b not in only[pre]: continue
        if path.startswith("REPO:"): p = D / b / "features" / path[5:]
        else:
            if pre in ("plan3", "dry") and "evo_check" in path:
                if b != "tauk_banking": continue
            elif "evo_check" not in path and b == "tauk_banking" and pre in ("plan3", "dry") and False: pass
            p = S / path.format(b=b)
        if not p.exists():
            skipped.append((pre, str(path), b)); continue
        d = pd.read_csv(p)
        if pre == "emu" and "case_id" in d: d = d.drop(columns=["procedure_id"], errors="ignore")
        idc = "case_id" if "case_id" in d else d.columns[0]
        d = d.set_index(idc)
        d = d[~d.index.duplicated()]
        d = d.select_dtypes(include=[np.number, bool]).astype(float)
        if len(d.index.intersection(t.index)) < 0.9 * len(t):
            skipped.append((pre, str(p), b + " idmismatch")); continue
        for c in d.columns:
            if pre in ("plan3", "dry") and "evo_check" in str(p) and False: pass
            name = f"{pre}.{c}"
            if name in feats: continue
            feats[name] = d[c].reindex(t.index); meta[name] = (fam, trk)
        got.append((pre, str(p)))
    # surface (keyed by bench/case)
    sf = pd.read_csv(S / "ideas8/surface/features.csv")
    sf = sf[sf.bench == b].set_index("case")
    if len(sf):
        for c in ["icomp_lzma", "icomp_abs", "icomp_zlib", "comp_req", "fert", "ntok", "ndigtok", "fdigtok", "nnum", "nid", "ovl", "novl", "ncw"]:
            feats[f"surf.{c}"] = sf[c].reindex(t.index); meta[f"surf.{c}"] = ("SURFACE", "L0")
    # procedure-level compile unit features
    if b == "sopbench":
        u = pd.read_csv(S / "ideas3/compile_consistency/unit_features.csv")
        u["procedure_id"] = u.unit.str.rsplit("/", n=1).str[0]
        g = u.drop(columns="unit").groupby("procedure_id").mean()
        for c in g.columns:
            feats[f"unit.{c}"] = t.procedure_id.map(g[c]); meta[f"unit.{c}"] = ("CMP_UNIT(proc-level)", "L1")
    X = pd.DataFrame(feats)
    return t, X, meta, got, skipped

def cen(v, g):
    return v - v.groupby(g).transform("mean")

def sp(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 8: return np.nan
    x, y = x[m], y[m]
    if np.ptp(x) == 0 or np.ptp(y) == 0: return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])

def within_rho(x, b, g, dom, mask_dom):
    """Spearman of procedure-centred x and b in one domain (rows of non-singleton procedures)."""
    xs, bs, gs = x[mask_dom], b[mask_dom], g[mask_dom]
    n = gs.map(gs.value_counts())
    k = (n >= 2) & xs.notna()
    xs, bs, gs = xs[k], bs[k], gs[k]
    if len(xs) < 8: return np.nan
    return sp(cen(xs, gs).values, cen(bs, gs).values)

def main():
    rng = np.random.default_rng(0)
    rows = []; allmeta = {}; log = {}
    T, XX = {}, {}
    for b in BENCH:
        t, X, meta, got, skipped = load_bench(b)
        T[b], XX[b] = t, X; allmeta.update(meta); log[b] = (got, skipped)
    feats = sorted(set().union(*[set(XX[b].columns) for b in BENCH]))
    NPERM = 200
    res = []
    for b in BENCH:
        t, X = T[b], XX[b]
        g = t.procedure_id; bb = t.b
        domains = sorted(t.domain.unique())
        dm = {d: (t.domain == d).values for d in domains}
        multi = g.map(g.value_counts()) >= 2
        # null distribution: shuffle b within procedure
        perm_rho = {}  # (feat, dom) -> array of perms
        if b != "tauk_banking":
            gidx = g.groupby(g).indices
            Xc = {f: cen(X[f], g) for f in X.columns}
            perm_b = []
            for _ in range(NPERM):
                nb = bb.values.copy()
                for ix in gidx.values():
                    if len(ix) > 1: nb[ix] = nb[rng.permutation(ix)]
                perm_b.append(pd.Series(nb, index=bb.index))
        for f in X.columns:
            x = X[f]
            if x.notna().sum() < 20: continue
            r = dict(bench=b, feat=f, family=allmeta[f][0], track=allmeta[f][1])
            r["n"] = int(x.notna().sum())
            r["nuniq"] = int(x.nunique())
            # raw within-domain
            raw = {d: sp(x[dm[d]].values, bb[dm[d]].values) for d in domains}
            r["raw"] = raw
            if b == "tauk_banking":
                res.append(r); continue
            wi = {d: within_rho(x, bb, g, d, dm[d]) for d in domains}
            r["within"] = wi
            # within by procedure x label (sopbench)
            if b == "sopbench":
                gl = g + "|" + t.should_succeed.astype(str)
                wl = {}
                for d in domains:
                    wl[d] = within_rho(x, bb, gl, d, dm[d])
                r["within_label"] = wl
            # variance share within procedure (rows in multi-case procedures)
            xm = x[multi & x.notna()]
            if len(xm) > 5 and xm.var() > 0:
                gm = g[xm.index]
                r["var_within"] = float(cen(xm, gm).pow(2).sum() / ((xm - xm.mean()).pow(2).sum()))
            else: r["var_within"] = np.nan
            r["zero_within"] = bool(np.nan_to_num(r["var_within"], nan=0.0) < 1e-8)
            # between-procedure Spearman (procedure means) per domain
            pm = pd.DataFrame({"x": x, "b": bb, "g": g, "d": t.domain}).groupby("g").agg(x=("x", "mean"), b=("b", "mean"), d=("d", "first"))
            r["between"] = {d: sp(pm[pm.d == d].x.values, pm[pm.d == d].b.values) for d in domains}
            # FE slope with cluster-robust z (within-procedure), standardised x
            ok = multi & x.notna()
            xs = x[ok]; sd = x.std()
            if r["zero_within"] or not sd > 0:
                r["fe_slope"] = np.nan; r["fe_z"] = np.nan
            else:
                xc = cen(xs / sd, g[ok]).values; yc = cen(bb[ok], g[ok]).values
                sxx = (xc ** 2).sum(); beta = (xc * yc).sum() / sxx
                e = yc - beta * xc
                gi = g[ok].values
                sc = pd.Series(xc * e).groupby(gi).sum()
                G = len(sc); n_ = ok.sum()
                var = (sc ** 2).sum() / sxx ** 2 * (G / (G - 1)) * ((n_ - 1) / max(n_ - 1 - 1, 1))
                r["fe_slope"] = float(beta); r["fe_z"] = float(beta / np.sqrt(var)) if var > 0 else np.nan
                # between slope z (OLS on procedure means), standardised
                pmx = pm.dropna(); 
                if pmx.x.std() > 0:
                    zz = (pmx.x - pmx.x.mean()) / pmx.x.std()
                    sl = np.polyfit(zz, pmx.b, 1)[0]
                    res_ = pmx.b - np.polyval(np.polyfit(zz, pmx.b, 1), zz)
                    se = np.sqrt((res_ ** 2).sum() / (len(pmx) - 2) / (zz ** 2).sum())
                    r["btw_slope"] = float(sl); r["btw_z"] = float(sl / se)
            # permutation null of the within rho per domain, and for the bench mean
            nullm = []; nulld = {d: [] for d in domains}
            for pb in perm_b:
                vals = []
                for d in domains:
                    v = within_rho(x, pb, g, d, dm[d]); nulld[d].append(v); vals.append(v)
                nullm.append(np.nanmean(vals) if not np.all(np.isnan(vals)) else np.nan)
            r["null95_dom"] = {d: float(np.nanpercentile(np.abs(nulld[d]), 95)) if not np.all(np.isnan(nulld[d])) else np.nan for d in domains}
            r["null95_mean"] = float(np.nanpercentile(np.abs(nullm), 95)) if not np.all(np.isnan(nullm)) else np.nan
            res.append(r)
    pd.to_pickle(res, OUT / "res.pkl")
    pd.to_pickle(log, OUT / "log.pkl")
    print("done", len(res))

if __name__ == "__main__":
    with cpu_lock("diag"):
        main()
