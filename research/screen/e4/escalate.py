"""E4, escalation curve (PREREG.md): an agent handles the incoming new tasks; the share x it is least likely to solve
(by predicted P) goes to a human (assumed correct) or to the strongest agent. Gain over random escalation, per method.
Out-of-fold P from ../ideas10/fc/out (same folds as the baseline protocol)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json
import numpy as np, pandas as pd

OBS = ZR.SCREEN + "/ideas10/fc/out/{}/obs_{}.csv.gz"
METHODS = ["adele", "c1_all", "f2_all_fc", "oracle"]
XS = [0.1, 0.2, 0.3]


def cells(bench, scheme):
    o = pd.read_csv(OBS.format(bench, scheme), usecols=["seed", "fold", "agent", "case_id", "y"] + METHODS)
    g = o.groupby(["seed", "agent", "case_id"]).agg(y=("y", "mean"), **{m: (m, "first") for m in METHODS}).reset_index()
    g["proc"] = g.case_id.str.rsplit("/", n=1).str[0] if bench == "sopbench" else g.case_id
    return g


def curve(df, score, x, to):
    """mean success when the share x of tasks with the lowest predicted P goes to `to` (array of success values or 1.0)"""
    n = len(df); k = int(round(x * n))
    order = np.argsort(df[score].values, kind="stable")      # lowest P first
    esc = np.zeros(n, bool); esc[order[:k]] = True
    tgt = to if np.ndim(to) else np.full(n, to)
    return float(np.where(esc, tgt, df.y.values).mean())


def run(bench, scheme, B=300):
    g = cells(bench, scheme)
    best = g.groupby("agent").y.mean().idxmax()
    res = {"best_agent": best}
    rows = []
    for seed, gs in g.groupby("seed"):
        strong = gs[gs.agent == best].set_index("case_id").y
        for a, ga in gs.groupby("agent"):
            if a == best: continue
            ga = ga.reset_index(drop=True)
            base = ga.y.mean()
            for x in XS:
                for to_name, to in (("human", 1.0), ("best agent", strong.loc[ga.case_id].values)):
                    rnd = base + x * ((to if np.ndim(to) == 0 else to.mean()) - base)
                    for m in METHODS:
                        rows.append({"seed": seed, "agent": a, "x": x, "to": to_name, "method": m, "gain": curve(ga, m, x, to) - rnd})
    r = pd.DataFrame(rows)
    tab = r.groupby(["to", "x", "method"]).gain.mean().unstack("method")
    res["table"] = {f"{to}|{x}": tab.loc[(to, x)].to_dict() for to, x in tab.index}
    # paired bootstrap over procedures for f2 minus baseline gain at x = 0.2 (human), recomputed on resampled task sets
    rng = np.random.default_rng(0); procs = g.proc.unique()
    diffs = []
    for _ in range(B):
        w = pd.Series(rng.choice(procs, len(procs))).value_counts()
        sub = g[g.proc.isin(w.index)]
        sub = sub.loc[sub.index.repeat(sub.proc.map(w).astype(int))]
        d = []
        for seed, gs in sub.groupby("seed"):
            for a, ga in gs.groupby("agent"):
                if a == best: continue
                ga = ga.reset_index(drop=True)
                d.append(curve(ga, "f2_all_fc", 0.2, 1.0) - curve(ga, "adele", 0.2, 1.0))
        diffs.append(np.mean(d))
    res["f2_minus_adele_human_0.2"] = {"diff": float(tab.loc[("human", 0.2), "f2_all_fc"] - tab.loc[("human", 0.2), "adele"]),
                                       "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    return res


if __name__ == "__main__":
    out = {}
    for bench in ("sopbench", "tau2"):
        for scheme in ("new_procedures", "new_domain"):
            out[f"{bench}|{scheme}"] = r = run(bench, scheme)
            print("==", bench, scheme, "best agent", r["best_agent"])
            for k, v in r["table"].items(): print("  ", k.ljust(16), {m: round(x * 100, 1) for m, x in v.items()})
            print("   f2 - adele (human, 20%), points:", round(r["f2_minus_adele_human_0.2"]["diff"] * 100, 1), [round(c * 100, 1) for c in r["f2_minus_adele_human_0.2"]["ci"]])
    json.dump(out, open(ZR.SCREEN + "/e4/escalate.json", "w"), indent=1)
