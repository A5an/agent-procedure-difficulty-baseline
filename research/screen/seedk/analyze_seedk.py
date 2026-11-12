"""E1 analysis (PREREG.md), written before the full episodes finished. E2 (../sim2) is reported descriptively, since the
70 USD total stopped it at about a third. No model calls.

Per agent and condition: failure rate of the probes per family vs real failure rate per family (6 real tasks), Spearman
over families, bootstrap over families (2000, seed 20261112), paired differences on the same resamples.
k0 = documentation probes, k2 = seeded probes (both recorded earlier), k1 / k5 / foreign = this folder. k1 exists for
28 families only (generation stopped by the budget), so its comparisons are on those 28.

  python3 analyze_seedk.py  -> analyze_seedk.out (stdout), results_seedk.json"""
import json, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; SCR = HERE.parent
SEL = json.load(open(HERE / "selection_seedk.json")); F = sorted(SEL["families"])
AG = ["g38", "flash"]
FILES = {"real": ["tib/runs/real_{a}.jsonl", "combo_rep/runs/real_{a}.jsonl"],
         "k0": ["tib/runs/probes_{a}.v2.jsonl", "combo_rep/runs/doc_{a}.jsonl"],
         "k2": ["tib/runs/seeded_{a}.jsonl", "combo_rep/runs/seeded_{a}.jsonl"],
         "k1": ["seedk/runs/k1_{a}.jsonl"], "k5": ["seedk/runs/k5_{a}.jsonl"], "foreign": ["seedk/runs/foreign_{a}.jsonl"],
         "k2_pro": ["sim2/runs/seeded_pro_{a}.jsonl"]}
COND = ["k0", "k1", "k2", "k5", "foreign"]
B, SEED = 2000, 20261112


def fam_of(tid):
    return next(f for f in F if f"_{f}_" in tid)


def load(src, a):
    succ, n, per = collections.Counter(), collections.Counter(), {}
    seen = set()
    for p in FILES[src]:
        path = SCR / p.format(a=a)
        if not path.exists(): continue
        for l in open(path):
            r = json.loads(l)
            if r.get("reward") is None or r["task_id"] in seen: continue
            seen.add(r["task_id"]); f = fam_of(r["task_id"])
            n[f] += 1; succ[f] += int(r["reward"] >= 1.0); per[r["task_id"]] = int(r["reward"] >= 1.0)
    return succ, n, per


def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v; r[m] = r[m].mean()
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0: return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]


OUT = {}
rng = np.random.default_rng(SEED)
for a in AG:
    sR, nR, _ = load("real", a)
    real_fail = {f: 1 - sR[f] / nR[f] for f in F}
    data = {c: load(c, a) for c in COND + ["k2_pro"]}
    full = {c: [f for f in F if data[c][1][f] >= 1] for c in COND}
    print(f"\n=== agent {a}: real success {sum(sR.values()) / sum(nR.values()):.3f} over {sum(nR.values())} cases")
    for c in COND:
        s, n, _ = data[c]
        print(f"  {c:8s} families {len(full[c]):2d}, probes {sum(n.values()):4d}, probe success {sum(s.values()) / max(1, sum(n.values())):.3f}"
              + ("" if all(n[f] == 6 for f in full[c]) else f", families with fewer than 6 probes: {sum(n[f] < 6 for f in full[c])}"))
    pf = {c: {f: 1 - data[c][0][f] / data[c][1][f] for f in full[c]} for c in COND}

    def rho_on(c, fams, idx=None):
        fams = fams if idx is None else [fams[i] for i in idx]
        return spearman([pf[c][f] for f in fams], [real_fail[f] for f in fams])

    sets = {"all50": [f for f in F if all(f in pf[c] for c in ("k0", "k2", "k5", "foreign"))],
            "k1_28": [f for f in F if all(f in pf[c] for c in COND)]}
    res = {}
    for sname, fams in sets.items():
        conds = [c for c in COND if all(f in pf[c] for f in fams)]
        boots = {c: [] for c in conds}
        for _ in range(B):
            idx = rng.integers(0, len(fams), len(fams))
            for c in conds: boots[c].append(rho_on(c, fams, idx))
        point = {c: rho_on(c, fams) for c in conds}
        print(f"  -- families: {sname} ({len(fams)})")
        for c in conds:
            print(f"     rho {c:8s} {point[c]:+.3f}  [{ci(boots[c])[0]:+.3f}, {ci(boots[c])[1]:+.3f}]")
        pairs = [("k5", "k0"), ("k2", "k0"), ("k5", "k2"), ("k2", "foreign"), ("foreign", "k0")] + (
            [("k1", "k0"), ("k2", "k1"), ("k5", "k1")] if "k1" in conds else [])
        diffs = {}
        for x, y in pairs:
            d = np.array(boots[x]) - np.array(boots[y])
            diffs[f"{x}-{y}"] = {"point": point[x] - point[y], "ci": ci(d)}
            print(f"     diff {x}-{y:8s} {point[x] - point[y]:+.3f}  [{ci(d)[0]:+.3f}, {ci(d)[1]:+.3f}]")
        mono = all(point[c1] <= point[c2] for c1, c2 in zip([c for c in ("k0", "k1", "k2", "k5") if c in point],
                                                           [c for c in ("k0", "k1", "k2", "k5") if c in point][1:]))
        print(f"     monotone in k: {mono}")
        res[sname] = {"n_families": len(fams), "rho": {c: {"point": point[c], "ci": ci(boots[c])} for c in conds},
                      "diffs": diffs, "monotone": mono}
    # probes passed of 6 -> real success
    table = {}
    for c in COND:
        rows = collections.defaultdict(lambda: [0, 0, 0])
        for f in full[c]:
            k = data[c][0][f]; rows[k][0] += 1; rows[k][1] += sR[f]; rows[k][2] += nR[f]
        table[c] = {int(k): {"families": v[0], "real": f"{v[1]}/{v[2]}", "rate": v[1] / v[2]} for k, v in sorted(rows.items(), reverse=True)}
        print(f"  {c:8s} probes passed -> real success: " + "; ".join(f"{k}: {v['families']} fam {v['real']} = {v['rate']:.2f}" for k, v in table[c].items()))
    # Brier of the raw probe rate as a forecast of each real case
    brier = {}
    for c in COND:
        e, m = 0.0, 0
        for f in full[c]:
            p = data[c][0][f] / data[c][1][f]
            e += sR[f] * (1 - p) ** 2 + (nR[f] - sR[f]) * p ** 2; m += nR[f]
        brier[c] = e / m
    print("  Brier (raw probe success rate as the forecast of each real case): " + ", ".join(f"{c} {v:.4f}" for c, v in brier.items()))
    # E2, descriptive: probes run with both simulators
    s2, n2, per2 = data["k2_pro"]; _, _, per1 = data["k2"]
    both = [t for t in per2 if t in per1]
    dis = sum(per1[t] != per2[t] for t in both)
    fams2 = [f for f in F if n2[f] == 6]
    e2 = {"probes_both": len(both), "disagree": dis, "pass_flash_sim": sum(per1[t] for t in both), "pass_pro_sim": sum(per2[t] for t in both),
          "families_complete": len(fams2)}
    if len(fams2) >= 5:
        e2["rho_flash_sim_on_these"] = spearman([pf["k2"][f] for f in fams2], [real_fail[f] for f in fams2])
        e2["rho_pro_sim_on_these"] = spearman([1 - s2[f] / 6 for f in fams2], [real_fail[f] for f in fams2])
    print(f"  E2 (partial): {e2}")
    OUT[a] = {"sets": res, "passed_table": table, "brier": brier, "e2_partial": e2}
json.dump(OUT, open(HERE / "results_seedk.json", "w"), indent=1, default=float)
