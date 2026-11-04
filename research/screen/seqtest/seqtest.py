"""How many probes does a delegation decision need? Sequential rule on SOPBench gemini-3.8-flash probes (about 20 per
procedure, recorded). Exploratory, no API calls. Rule: Beta prior fitted by moments on the other domains' probe rates;
run probes one by one in random order; delegate when P(success rate >= t) >= c, refuse when P(rate < t) >= c, else go
on; at the end of the probes decide by the posterior mean >= t. Compared with fixed n = 6 and all probes."""
import csv, collections, numpy as np
from pathlib import Path
from scipy.stats import beta as B
G = Path(__file__).resolve().parent.parent / "ideas5/smart_gemini"
real, pr = collections.defaultdict(list), collections.defaultdict(list)
for r in csv.DictReader(open(G / "real_results.csv")): real["/".join(r["case_id"].split("/")[:2])].append(int(r["success"]))
for r in csv.DictReader(open(G / "synth_results.csv")): pr[r["procedure_id"]].append(int(r["success"]))
procs = sorted(p for p in real if p in pr)
rng = np.random.default_rng(3)
for t in (0.9, 0.8):
    for c in (0.8, 0.9):
        res = collections.defaultdict(list)
        for rep in range(50):
            for p in procs:
                o = [q for q in procs if q.split("/")[0] != p.split("/")[0]]
                rates = np.array([np.mean(pr[q]) for q in o]); m, v = rates.mean(), max(rates.var(), 1e-4); k0 = max(m * (1 - m) / v - 1, .5)
                a0, b0 = m * k0, (1 - m) * k0
                seq = rng.permutation(pr[p]); s = f = 0; dec = None
                for i, x in enumerate(seq, 1):
                    s += x; f += 1 - x
                    up = 1 - B.cdf(t, a0 + s, b0 + f)
                    if up >= c: dec = True; break
                    if 1 - up >= c: dec = False; break
                if dec is None: dec = (a0 + s) / (a0 + b0 + s + f) >= t
                res["seq"].append((p, dec, i))
                for n in (6, len(seq)):
                    ss = seq[:n].sum(); res[f"fixed{'6' if n == 6 else 'all'}"].append((p, (a0 + ss) / (a0 + b0 + n) >= t, n))
        print(f"\nthreshold t={t}, confidence c={c}")
        for k, L in res.items():
            dl = [p for p, d, n in L if d]
            cases = sum(len(real[p]) for p, d, n in L) ; dc = sum(len(real[p]) for p in dl)
            succ = np.mean([x for p in dl for x in real[p]]) if dl else float("nan")
            print(f"  {k:9s} probes per procedure {np.mean([n for _, _, n in L]):5.1f}  delegated {dc / cases:4.0%} of cases  success among delegated {succ:4.0%}")
