"""Analysis of the synthetic-procedure experiment (PREREG.md): generators G1-G7, factors H1-H6, monitor transfer.

  python analyze.py  -> prints tables, writes results.json
"""
import json, re, itertools, collections, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
import dataset, engine

HERE = Path(__file__).resolve().parent
GENS, AGENTS = ["haiku", "sonnet", "opus"], ["lite", "flash", "g38"]
BRIEFS = dataset.BRIEFS
PRE = f"{dataset.V}_" if dataset.V else ""  # hard round: SYNTH_V=v2
rng = np.random.default_rng(20261106)
B = 2000
R = {}


def boot_ci(df, stat, cluster="proc", B=B):
    """percentile interval of stat(df) over resamples of clusters"""
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = list(groups)
    vals = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys))
        d = pd.concat([groups[keys[i]] for i in pick], ignore_index=True)
        try:
            v = stat(d)
        except Exception:
            continue
        if v is not None and np.all(np.isfinite(v)):
            vals.append(v)
    vals = np.array(vals)
    if len(vals) < 20:
        return np.nan, np.nan
    return np.percentile(vals, 2.5, axis=0), np.percentile(vals, 97.5, axis=0)


def fmt(x, lo, hi, p=2):
    return f"{x:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


# ------------------------------------------------------------------ G1 validity and cost of generation
print("== G1 generation")
g1 = {}
for g in GENS:
    rows = []
    for pid in BRIEFS:
        lp = HERE / dataset.GEN_DIR / g / f"{pid}.log.jsonl"
        if not lp.exists():
            continue
        L = [json.loads(l) for l in open(lp)]
        s = json.load(open(HERE / dataset.GEN_DIR / g / f"{pid}.json"))
        rows.append({"pid": pid, "accepted": not s.get("rejected"), "round0_clean": L[0]["n_errors"] == 0,
                     "round0_proc_ok": L[0]["proc_ok"], "round0_cases_ok": L[0]["n_cases_ok"], "rounds": len(L),
                     "cases_kept": s.get("_check", {}).get("cases_ok", 0) if not s.get("rejected") else 0,
                     "sec": sum(x.get("sec", 0) for x in L), "out_tokens": sum(x.get("out") or 0 for x in L),
                     "notional_usd": sum(x.get("notional_usd") or 0 for x in L)})
    d = pd.DataFrame(rows)
    g1[g] = {"briefs": len(d), "accepted": int(d.accepted.sum()), "round0_clean": int(d.round0_clean.sum()),
             "round0_proc_ok": int(d.round0_proc_ok.sum()), "round0_cases_ok_mean": float(d.round0_cases_ok.mean()),
             "mean_rounds": float(d.rounds.mean()), "cases_kept": int(d.cases_kept.sum()),
             "minutes_per_brief": float(d.sec.mean() / 60), "out_tokens_per_brief": float(d.out_tokens.mean()),
             "notional_usd_total": float(d.notional_usd.sum())}
    print(g, g1[g])
R["G1"] = g1

# ------------------------------------------------------------------ G2 policy fidelity (judge)
print("== G2 policy check")
g2 = {}
for g in GENS:
    p = HERE / "judge" / f"{PRE}{g}.jsonl"
    if not p.exists():
        continue
    J = [json.loads(l) for l in open(p)]
    first = {j["pid"]: j for j in J if j["round"] == 0}
    last = {}
    for j in J:
        last[j["pid"]] = j
    def exact_share(js):
        v = [x.get("verdict") == "exact" for j in js for x in j.get("verdicts", [])]
        return float(np.mean(v)) if v else float("nan")
    g2[g] = {"procs": len(first), "clean_at_first": sum(j["ok"] for j in first.values()),
             "clean_after_fix": sum(j["ok"] for j in last.values()), "rules_exact_first": exact_share(first.values()),
             "rules_exact_final": exact_share(last.values()), "flagged_final": sorted(p for p, j in last.items() if not j["ok"])}
    print(g, {k: v for k, v in g2[g].items() if k != "flagged_final"}, "flagged:", g2[g]["flagged_final"])
R["G2"] = g2
FLAGGED = {(g, p) for g in g2 for p in g2[g]["flagged_final"]}

# ------------------------------------------------------------------ cases, features, episodes
items = {}
for g in GENS:
    for it in dataset.cases(g):
        items[it["uid"]] = it
rows = []
for g in GENS:
    for a in AGENTS:
        p = HERE / "runs" / f"{PRE}{g}_{a}.jsonl"
        if not p.exists():
            continue
        last = {}
        for l in open(p):
            r = json.loads(l)
            last[r["task_id"]] = r
        for uid, r in last.items():
            if r.get("reward") is None or uid not in items:
                continue
            it = items[uid]
            st = BRIEFS[it["pid"]]["structure"]
            rows.append({"uid": uid, "gen": g, "agent": a, "proc": f"{g}/{it['pid']}", "pid": it["pid"], "case": it["case"]["case_id"],
                         "fail": 1 - int(r["reward"] >= 1), "level": it["plan"]["level"], "behavior": it["plan"]["behavior"],
                         "cls": it["sol"]["cls"], "n_rules": st["n_rules"], "n_calls": st["n_calls"],
                         "escalation": int(st["escalation"]), "verification": int(st["verification"]),
                         "flagged": int((g, it["pid"]) in FLAGGED), "domain": BRIEFS[it["pid"]]["domain"],
                         **{f"f_{k}": v for k, v in it["feat"].items() if isinstance(v, (int, float, bool))}})
E = pd.DataFrame(rows)
print("episodes:", len(E), E.groupby(["gen", "agent"]).size().unstack().to_dict() if len(E) else "")
if E.empty:
    json.dump(R, open(HERE / f"{PRE}results.json", "w"), indent=1, default=str)
    sys.exit()

# ------------------------------------------------------------------ G3 diversity
print("== G3 diversity")
ex = json.load(open(HERE / "example_procedure.json"))


def grams(t):
    w = re.findall(r"[a-z0-9]+", t.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


exg = grams(ex["policy_text"])
g3 = {}
for g in GENS:
    specs = dataset.load_specs(g)
    if not specs:
        continue
    pol = [grams(s["policy_text"]) for s in specs.values()]
    per = [grams(" ".join(c["persona"] for c in s["cases"])) for s in specs.values()]
    g3[g] = {"entities": float(np.mean([len(s["entities"]) for s in specs.values()])),
             "tools": float(np.mean([len(s["tools"]) for s in specs.values()])),
             "search_tools": float(np.mean([sum(t["kind"] == "search" for t in s["tools"]) for s in specs.values()])),
             "bindings": float(np.mean([len(s["bindings"]) for s in specs.values()])),
             "fields_per_entity": float(np.mean([np.mean([len(e["fields"]) for e in s["entities"].values()]) for s in specs.values()])),
             "policy_words": float(np.mean([len(s["policy_text"].split()) for s in specs.values()])),
             "policy_3gram_jaccard_within": float(np.mean([jac(a, b) for a, b in itertools.combinations(pol, 2)])),
             "persona_3gram_jaccard_within": float(np.mean([jac(a, b) for a, b in itertools.combinations(per, 2)])),
             "policy_3gram_jaccard_example": float(np.mean([jac(a, exg) for a in pol])),
             "rules_deciding_some_case": float(np.mean([len({it["sol"]["decided_by"] for it in items.values() if it["gen"] == g and it["pid"] == pid} & {r["id"] for r in s["rules"]}) / len(s["rules"]) for pid, s in specs.items()]))}
    print(g, {k: round(v, 3) for k, v in g3[g].items()})
S3 = {g: dataset.load_specs(g) for g in GENS}
for a, b in itertools.combinations(GENS, 2):
    common = sorted(set(S3[a]) & set(S3[b]))
    pj = np.mean([jac(grams(S3[a][p]["policy_text"]), grams(S3[b][p]["policy_text"])) for p in common])
    tw = lambda s: set(re.findall(r"[a-z]+", s["title"].lower())) - {"and", "of", "the", "a", "for", "to", "on"}
    tj = np.mean([len(tw(S3[a][p]) & tw(S3[b][p])) / max(1, len(tw(S3[a][p]) | tw(S3[b][p]))) for p in common])
    g3[f"{a}~{b}"] = {"same_brief_policy_3gram_jaccard": float(pj), "same_brief_title_word_jaccard": float(tj)}
    print(f"{a} vs {b} on the same brief: policy 3-gram Jaccard {pj:.3f}, title word Jaccard {tj:.2f}")
R["G3"] = g3

# ------------------------------------------------------------------ G4 difficulty, G5 discrimination
print("== G4/G5 difficulty and discrimination")
g45 = {}
for g in GENS:
    for a in AGENTS:
        d = E[(E.gen == g) & (E.agent == a)]
        if d.empty:
            continue
        m = d.fail.mean(); lo, hi = boot_ci(d, lambda x: x.fail.mean())
        pr = d.groupby("pid").fail.mean()
        g45[f"{g}/{a}"] = {"n": len(d), "procs": int(d.pid.nunique()), "fail": float(m), "ci": [float(lo), float(hi)],
                           "sd_proc_fail": float(pr.std())}
        print(f"{g:7s} {a:6s} n {len(d):4d}  fail {fmt(m, lo, hi)}  SD of procedure rates {pr.std():.2f}")


def split_half(d):
    """odd vs even cases within procedure; procedure difficulty = mean failure over agents present"""
    d = d.copy()
    d["k"] = d.case.str[1:].astype(int) % 2
    t = d.groupby(["pid", "k"]).fail.mean().unstack().dropna()
    if len(t) < 5 or t[0].std() == 0 or t[1].std() == 0:
        return np.nan
    r = spearmanr(t[0], t[1]).correlation
    return 2 * r / (1 + r)


for g in GENS:
    d = E[E.gen == g]
    if d.empty:
        continue
    common = d.groupby("uid").agent.nunique()
    d = d[d.uid.isin(common[common == common.max()].index)]
    rel = split_half(d)
    lo, hi = boot_ci(d, split_half, cluster="pid", B=1000)
    order = d.groupby("agent").fail.mean().to_dict()
    g45[f"{g}/reliability"] = {"split_half_sb": float(rel), "ci": [float(lo), float(hi)], "agents": sorted(d.agent.unique()),
                               "fail_by_agent": order}
    print(f"{g:7s} split-half reliability of procedure difficulty {fmt(rel, lo, hi)}  agents {sorted(d.agent.unique())}  fail {order}")
R["G4G5"] = g45

# ------------------------------------------------------------------ G6 label audit
ap = HERE / f"{PRE}audit.jsonl"
if ap.exists():
    print("== G6 label audit")
    A = pd.DataFrame([json.loads(l) for l in open(ap)])
    tab = A.groupby(["gen", "cause"]).size().unstack(fill_value=0)
    print(tab)
    R["G6"] = tab.to_dict(orient="index")

# ------------------------------------------------------------------ G7 agreement between generators, factors
print("== per-level failure rates (rows: level, cols: generator), per agent")
lev_rates = {}
for a in AGENTS:
    d = E[E.agent == a]
    if d.empty:
        continue
    t = d.groupby(["level", "gen"]).fail.mean().unstack()
    lev_rates[a] = t.to_dict()
    print(a); print(t.round(2))
    for g1_, g2_ in itertools.combinations([g for g in GENS if g in t.columns], 2):
        x = t[[g1_, g2_]].dropna()
        if len(x) > 3:
            print(f"   level-profile correlation {g1_} vs {g2_}: {spearmanr(x[g1_], x[g2_]).correlation:.2f}")
R["G7_level_rates"] = lev_rates

LEVEL_REF, BEH_REF = "upfront", "cooperative"
LEVS = [l for l in engine.LEVELS if l != LEVEL_REF]
BEHS = [b for b in (engine.BEHAVIORS_V2 if dataset.V else engine.BEHAVIORS) if b != BEH_REF]


def design(d, with_gen):
    X = pd.DataFrame(index=d.index)
    for l in LEVS:
        X[f"level={l}"] = (d.level == l).astype(float)
    for b in BEHS:
        X[f"beh={b}"] = (d.behavior == b).astype(float)
    X["cls=perform"] = (d.cls == "perform").astype(float)
    X["n_rules"] = d.n_rules - 4
    X["n_calls"] = d.n_calls - 2
    X["escalation"] = d.escalation
    X["verification"] = d.verification
    if with_gen:
        for g in GENS[1:]:
            X[f"gen={g}"] = (d.gen == g).astype(float)
    return X


def fit(d, with_gen=True):
    X = design(d, with_gen)
    m = LogisticRegression(C=1.0, max_iter=5000).fit(X.values, d.fail.values)  # default L2: some levels are near-separable in resamples
    return m.coef_[0]


print("== factor models (logistic coefficients, log-odds; cluster bootstrap over procedures)")
fac = {}
for a in AGENTS:
    for scope in ["pooled"] + GENS:
        d = E[E.agent == a] if scope == "pooled" else E[(E.agent == a) & (E.gen == scope)]
        if d.empty or d.fail.nunique() < 2:
            continue
        cols = list(design(d.head(1), scope == "pooled").columns)
        est = fit(d, scope == "pooled")
        lo, hi = boot_ci(d, lambda x: fit(x, scope == "pooled"), B=B if scope == "pooled" else 1000)
        fac[f"{a}/{scope}"] = {c: [float(e), float(l), float(h)] for c, e, l, h in zip(cols, est, lo, hi)}
        if scope == "pooled":
            print(f"-- {a} pooled (n {len(d)})")
            for c, e, l, h in zip(cols, est, lo, hi):
                print(f"   {c:24s} {fmt(e, l, h)}{'  *' if l > 0 or h < 0 else ''}")
R["factors"] = fac

print("== raw failure rate by level and agent (pooled)")
print(E.groupby(["level", "agent"]).fail.mean().unstack().round(2))
print(E.groupby(["behavior", "agent"]).fail.mean().unstack().round(2))
R["raw_level"] = E.groupby(["level", "agent"]).fail.mean().unstack().to_dict()
R["raw_behavior"] = E.groupby(["behavior", "agent"]).fail.mean().unstack().to_dict()

# H5: lite minus g38 gap by level
if {"lite", "g38"} <= set(E.agent):
    w = E[E.agent.isin(["lite", "g38"])].pivot_table(index=["uid", "proc", "level"], columns="agent", values="fail").dropna().reset_index()
    def gapdiff(d, lev):
        return (d[d.level == lev].lite - d[d.level == lev].g38).mean() - (d[d.level == LEVEL_REF].lite - d[d.level == LEVEL_REF].g38).mean()
    R["H5"] = {}
    for lev in ("distracting", "conflicting"):
        v = gapdiff(w, lev); lo, hi = boot_ci(w, lambda d: gapdiff(d, lev))
        R["H5"][lev] = [float(v), float(lo), float(hi)]
        print(f"H5 gap(lite-g38) on {lev} minus on upfront: {fmt(v, lo, hi)}")

# robustness: without procedures the judge still flagged
if E.flagged.any():
    print("== pooled factor coefficients without flagged procedures")
    R["factors_unflagged"] = {}
    for a in AGENTS:
        d = E[(E.agent == a) & (E.flagged == 0)]
        if d.empty:
            continue
        cols = list(design(d.head(1), True).columns); est = fit(d)
        R["factors_unflagged"][a] = dict(zip(cols, map(float, est)))
        print(a, {c: round(float(e), 2) for c, e in zip(cols, est) if c.startswith("level")})

E.to_csv(HERE / f"{PRE}episodes.csv", index=False)
json.dump(R, open(HERE / f"{PRE}results.json", "w"), indent=1, default=str)
