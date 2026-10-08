"""Round 10 zs: c_lad (+GAP, +PRIOR) with all inputs z-scored within domain. usage: run.py"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys
S = ZR.SCREEN
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric"); sys.path.insert(0, S + "/ideas9/gap")
from pathlib import Path
import numpy as np, pandas as pd
import harness as H
from cpulock import cpu_lock
from ladder import Ladder
import prompts as P
from feats import GAP
import importlib.util
spec = importlib.util.spec_from_file_location("pf", S + "/ideas10/prior/feats.py"); pf = importlib.util.module_from_spec(spec); spec.loader.exec_module(pf)
PRIOR = pf.PRIOR
I3 = Path(S + "/ideas3"); HERE = Path(__file__).parent; REPO = H.REPO
def tab(path):
    d = pd.read_csv(path); return d.set_index(d.columns[0])
def zdom(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    dom = df.index.to_series().str.split("/").str[0]
    m = df.groupby(dom.values).transform("mean"); s = df.groupby(dom.values).transform("std").replace(0, np.nan)
    return ((df - m) / s).fillna(0.0)
TYPES = [f"n_{t}" for t in P.STEP_TYPES]; GLOB = ["max_depth", "refusal_conditions", "needs_hidden_data", "p_refusal"]
def make_preds(src, M, bench):
    sop = bench == "sopbench"
    ad = tab(REPO / "data" / bench / "features" / "adele.csv")
    rub = tab(S + f"/rubric/features/rub_proc_{bench}.csv")
    plan = tab(I3 / f"plan_tools/features/plan3_{bench}.csv")
    cmp_ = tab(I3 / "compile_consistency/cons_sopbench.csv")[["len_mean", "branch_mean"]] if sop else None
    g = tab(S + f"/ideas9/gap/gap_{bench}.csv")
    pr = tab(S + f"/ideas10/prior/prior_{bench}.csv")[PRIOR]
    base = [ad, plan] + ([cmp_] if sop else [])
    scn = [] if sop else [tab(S + "/ideas4/user_scenario/scn_tau2.csv")]
    rubz = rub.copy(); cols = [c for c in TYPES + GLOB if c in rub]; rubz[cols] = zdom(rub[cols])
    def L(name, parts, z):
        ex = pd.concat(parts, axis=1)
        return Ladder(2, TYPES, GLOB, rubz if z else rub, extra_table=zdom(ex) if z else ex, name=name)
    return {"t0_clad": L("t0_clad", base, False),
            "g1_clad_gap": L("g1_clad_gap", base + [g[GAP]], False),
            "pr2_cladgap_prior": L("pr2_cladgap_prior", base + [g[GAP], pr], False),
            "z0_clad": L("z0_clad", base, True),
            "z1_cladgap": L("z1_cladgap", base + [g[GAP]], True),
            "z2_cladgap_prior": L("z2_cladgap_prior", base + [g[GAP], pr], True),
            "c1_all": L("c1_all", base + [g[GAP]] + scn + [pr], False),
            "c1z_all": L("c1z_all", base + [g[GAP]] + scn + [pr], True)}
if __name__ == "__main__":
    with cpu_lock("zs"):
        H.run_all(HERE / "out", make_preds, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
    print("run done")
