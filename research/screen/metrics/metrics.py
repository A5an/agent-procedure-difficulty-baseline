"""README.md: probability metrics on recorded data. TauIndianBank 50 families (g38, flash), SOPBench 68 procedures
(gemini-2.5-flash-lite, gemini-2.5-flash from ../ideas2, gemini-3.8-flash from ../ideas5), leave one group out."""
import json, csv, collections
import numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent.parent


def tib_load(base, fams, fname):
    d = collections.defaultdict(list)
    p = base / "runs" / fname
    for l in open(p):
        r = json.loads(l)
        if r.get("reward") is None: continue
        f = next((f for f in fams if f"_{f}_" in r["task_id"]), None)
        if f: d[f].append(int(r["reward"] >= 1))
    return d


def sets():
    out = {}
    old = json.load(open(H / "tib/selection.json"))["test_families"]
    new = json.load(open(H / "combo_rep/selection_rep.json"))["test_families"]
    for a in ("g38", "flash"):
        real, seed, doc = {}, {}, {}
        for base, fams, dname in ((H / "tib", old, f"probes_{a}.v2.jsonl"), (H / "combo_rep", new, f"doc_{a}.jsonl")):
            real.update(tib_load(base, fams, f"real_{a}.jsonl")); seed.update(tib_load(base, fams, f"seeded_{a}.jsonl"))
            doc.update(tib_load(base, fams, dname))
        out[f"TauIndianBank {a}"] = {"real": real, "group": {f: f for f in real}, "probes": {"past-conversation probes": seed, "documentation probes": doc}}
    # SOPBench
    for a, rr, sr in (("lite", "real_lite", "synth_lite"), ("flash", "real_flash", "synth_flash")):
        real, pr = collections.defaultdict(list), collections.defaultdict(list)
        for x in json.load(open(H / "ideas2/emulate_run/runs" / rr / "results.json")):
            real[f"{x['domain']}/{x['proc']}"].append(int(x["success"]))
        for x in json.load(open(H / "ideas2/emulate_run/runs" / sr / "results.json")):
            pr[f"{x['cid'].split('/')[0]}/{x['proc']}"].append(int(x["success"]))
        out[f"SOPBench {a}"] = {"real": dict(real), "group": {p: p.split("/")[0] for p in real}, "probes": {"code-generated probes": dict(pr)}}
    real, pr = collections.defaultdict(list), collections.defaultdict(list)
    G = H / "ideas5/smart_gemini"
    for r in csv.DictReader(open(G / "real_results.csv")):
        real["/".join(r["case_id"].split("/")[:2])].append(int(r["success"]))
    for r in csv.DictReader(open(G / "synth_results.csv")):
        pr[r["procedure_id"]].append(int(r["success"]))
    out["SOPBench g38"] = {"real": dict(real), "group": {p: p.split("/")[0] for p in real}, "probes": {"code-generated probes": dict(pr)}}
    return out


def evaluate(name, S):
    procs = sorted(p for p in S["real"] if all(p in v and v[p] for v in S["probes"].values()))
    grp = S["group"]
    y = {p: np.array(S["real"][p]) for p in procs}
    preds = {}
    base = {p: np.mean(np.concatenate([y[q] for q in procs if grp[q] != grp[p]])) for p in procs}
    preds["base rate (no probes)"] = base
    for pn, P in S["probes"].items():
        raw = {p: (sum(P[p]) + 0.5) / (len(P[p]) + 1) for p in procs}
        preds[f"{pn}, raw"] = raw
        cal = {}
        for p in procs:
            o = [q for q in procs if grp[q] != grp[p]]
            shift = np.mean(np.concatenate([y[q] for q in o])) - np.mean([raw[q] for q in o])
            cal[p] = float(np.clip(raw[p] + shift, 0.01, 0.99))
        preds[f"{pn}, level-corrected"] = cal
    oracle = {p: y[p].mean() for p in procs}
    n = sum(len(y[p]) for p in procs)
    print(f"\n=== {name}: {len(procs)} procedures, {n} real cases, real success {np.mean(np.concatenate(list(y.values()))):.2f}")
    print(f"  {'predictor':44s} {'Brier':>6s} {'skill':>6s} {'logloss':>7s} {'calib':>6s} {'ECE':>5s} | delegate if p>=0.95 / 0.9 / 0.8: share of cases, success among them")
    res = {}
    bb = None
    for k, pr in list(preds.items()) + [("oracle (true rates)", oracle)]:
        pc = np.concatenate([np.full(len(y[p]), np.clip(pr[p], 1e-3, 1 - 1e-3)) for p in procs])
        yc = np.concatenate([y[p] for p in procs])
        brier = np.mean((pc - yc) ** 2); ll = -np.mean(yc * np.log(pc) + (1 - yc) * np.log(1 - pc))
        if k.startswith("base"): bb = brier
        w = np.array([len(y[p]) for p in procs]); d = np.array([pr[p] - y[p].mean() for p in procs])
        calib = np.average(np.abs(d), weights=w)
        ps = np.array([pr[p] for p in procs]); bins = np.quantile(ps, np.linspace(0, 1, 6)); ece = 0
        for i in range(5):
            m = (ps >= bins[i]) & (ps <= bins[i + 1]) if i == 4 else (ps >= bins[i]) & (ps < bins[i + 1])
            if m.any(): ece += w[m].sum() / w.sum() * abs(np.average(ps[m], weights=w[m]) - np.average([y[p].mean() for p, mm in zip(procs, m) if mm], weights=w[m]))
        sel = []
        for t in (0.95, 0.9, 0.8):
            ch = [p for p in procs if pr[p] >= t]
            cov = sum(len(y[p]) for p in ch) / n
            succ = np.mean(np.concatenate([y[p] for p in ch])) if ch else float("nan")
            sel.append(f"{cov:4.0%} {succ:4.0%}" if ch else "  0%   - ")
        res[k] = dict(brier=brier, skill=1 - brier / bb, logloss=ll, calib=calib, ece=ece, select=sel)
        print(f"  {k:44s} {brier:6.3f} {1 - brier / bb:+6.2f} {ll:7.3f} {calib:6.3f} {ece:5.3f} | " + "  /  ".join(sel))
    return res


R = {k: evaluate(k, S) for k, S in sets().items()}
json.dump(R, open(Path(__file__).with_name("metrics.json"), "w"), indent=1, default=str)
