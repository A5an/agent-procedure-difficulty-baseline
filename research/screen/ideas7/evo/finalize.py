"""Pick at most 3 finalists from the archive using SEARCH-split information only, write FINALISTS.md. No held-out file is opened here.
Rule (fixed before looking at archive results): (1) take the 12 best archive candidates by search score; (2) re-score each with 5 fresh
fold shuffles (seeds 5000+) on the search split, rank by the mean of original and fresh score; (3) walk the ranking, accept a candidate if
fresh score > +0.005 and its max within-domain |r| against every accepted finalist's features is below 0.7; stop at 3."""
import sys, json
sys.path.insert(0, "."); import evaluator as EV
import numpy as np
arch = [json.loads(l) for l in open("archive.jsonl")]
ok = sorted([a for a in arch if a["status"] == "ok" and a["score"] is not None], key=lambda a: -a["score"])[:12]
t, rp = EV.load("search")
def fresh(code):
    o = EV.score(code, "search", return_matrix=True)
    tt = t.set_index("case_id").loc[o["ids"]].reset_index()
    F = o["matrix"]; dom = tt.domain.values; grp = tt.procedure_id.values
    y = EV.demean(tt.y_irt_b.values[:, None], dom)[:, 0]; b = EV.demean(tt.clad_oof.values[:, None], dom)[:, 0]; X = EV.demean(F, dom)
    gs = []
    for rep in range(5):
        rng = np.random.default_rng(5000 + rep); fo = EV.group_folds(grp, EV.OUTER, rng)
        pf, pb = np.zeros(len(y)), np.zeros(len(y))
        for i in range(EV.OUTER):
            tr, te = fo != i, fo == i
            pf[te], _ = EV.fit_pred(X[tr], y[tr], b[tr], X[te], b[te], grp[tr], rng)
            pb[te] = (b[tr] @ y[tr] / (b[tr] @ b[tr])) * b[te]
        gs.append(EV.within_rho(pf, y, dom)[0] - EV.within_rho(pb, y, dom)[0])
    return float(np.mean(gs)), X
rows = []
for a in ok:
    fs, X = fresh(a["code"]); rows.append((0.5 * (a["score"] + fs), fs, a, X))
rows.sort(key=lambda r: -r[0])
fin = []
for comb, fs, a, X in rows:
    if fs <= 0.005: continue
    Z = X / np.where(X.std(0) < 1e-9, 1, X.std(0))
    if all(np.abs(np.corrcoef(Z.T, f[3].T)[:Z.shape[1], Z.shape[1]:]).max() < 0.7 for f in fin): fin.append((comb, fs, a, Z))
    if len(fin) == 3: break
md = ["# FINALISTS (written before any held-out look)\n", "Selection rule: see finalize.py docstring. Search-split scores are biased upward (selected maxima of ~500 candidates).\n",
      "Top-12 table (id, island, search score, fresh-shuffle score, combined):\n"]
for comb, fs, a, X in rows: md.append(f"- id {a['id']} {a['island']} round {a['round']}: search {a['score']:+.4f}, fresh {fs:+.4f}, combined {comb:+.4f}")
md.append("\nPre-declared held-out and full-protocol evaluation (done once each after this file exists):\n- held-out A: same scoring as the search (ridge fitted by CV inside the held-out domains), gain over c_lad alone, per domain and mean over the 4 held-out domains (telecom reported separately, only 3 procedures)\n- held-out B (transfer): ridge fitted on all search cases, applied to held-out cases, same gain definition\n- full protocol harness: columns evo_fK (plain ladder with c_lad table + finalist features, domain-centred) and evo_fK_fam (per-family penalty ladder), paired CIs vs c_lad and adele; primary column = evo_fK\n")
for n, (comb, fs, a, Z) in enumerate(fin, 1):
    md.append(f"\n## Finalist {n}: archive id {a['id']} (island {a['island']}, round {a['round']}), search {a['score']:+.4f}, fresh {fs:+.4f}\nFeatures kept: {a['feature_names']}\nIdea: {a['idea']}\n```python\n{a['code']}\n```")
open("FINALISTS.md", "w").write("\n".join(md)); json.dump([{"n": n, "id": a["id"], "code": a["code"], "names": a["feature_names"], "search": a["score"], "fresh": fs} for n, (c, fs, a, Z) in enumerate(fin, 1)], open("finalists.json", "w"), indent=1)
print("\n".join(md[:20]))
