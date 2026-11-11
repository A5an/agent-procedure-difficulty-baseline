"""Exploratory, no API calls: can a cheap agent's probes give the expensive agent's percentage? 50 TauIndianBank
families. Target: g38 real success per family (and flash, the other way round). Predictors: own seeded probes;
the other agent's seeded probes (raw, and shifted by the level gap learned on the other families); both pooled."""
import json, collections, numpy as np
from pathlib import Path
S = Path(__file__).resolve().parent.parent
old, new = json.load(open(S / "tib/selection.json")), json.load(open(S / "combo_rep/selection_rep.json"))


def load(path, fams):
    d = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        if r.get("reward") is None: continue
        f = next((f for f in fams if f"_{f}_" in r["task_id"]), None)
        if f: d[f].append(int(r["reward"] >= 1))
    return d


D = {}
for a in ("g38", "flash"):
    real, seed = {}, {}
    for base, sel in ((S / "tib", old), (S / "combo_rep", new)):
        real.update(load(base / f"runs/real_{a}.jsonl", sel["test_families"])); seed.update(load(base / f"runs/seeded_{a}.jsonl", sel["test_families"]))
    D[a] = (real, seed)
fams = sorted(D["g38"][0])
for tgt, oth in (("g38", "flash"), ("flash", "g38")):
    real, own = D[tgt]; _, other = D[oth]
    y = {f: np.array(real[f]) for f in fams}
    r = lambda d, f: (sum(d[f]) + .5) / (len(d[f]) + 1)
    preds = {"own probes": {f: r(own, f) for f in fams}, f"{oth} probes, raw": {f: r(other, f) for f in fams}}
    sh = {}
    for f in fams:
        o = [g for g in fams if g != f]
        sh[f] = float(np.clip(r(other, f) + np.mean([y[g].mean() for g in o]) - np.mean([r(other, g) for g in o]), .01, .99))
    preds[f"{oth} probes, level-shifted"] = sh
    preds["both agents' probes pooled"] = {f: (sum(own[f]) + sum(other[f]) + .5) / (len(own[f]) + len(other[f]) + 1) for f in fams}
    base = {f: np.mean(np.concatenate([y[g] for g in fams if g != f])) for f in fams}
    yc = np.concatenate([y[f] for f in fams]); bb = np.mean((np.concatenate([np.full(6, base[f]) for f in fams]) - yc) ** 2)
    print(f"\n=== target {tgt} (50 families)")
    for k, p in preds.items():
        pc = np.concatenate([np.full(len(y[f]), p[f]) for f in fams]); br = np.mean((pc - yc) ** 2)
        ch = [f for f in fams if p[f] >= .9]
        cov = sum(len(y[f]) for f in ch) / len(yc); suc = np.mean(np.concatenate([y[f] for f in ch])) if ch else float("nan")
        print(f"  {k:34s} Brier skill {1 - br / bb:+.2f}  delegate p>=0.9: {cov:4.0%} of cases, success {suc:4.0%}")
