"""ready_proc for each seeds condition (plan of 13 Nov, item 3). The TauIndianBank part of ready_proc
(`../e5_embed/ready_proc_e5.py`, same split and qualification code, imported) with family similarity = bge without the
family-name prefix (E5), probe sources k0, k2, k5, foreign instead of doc/seeded, for the Gemini agents (g38, flash) and the
Claude agents (haiku, sonnet). The hierarchical-model row (mf) needs torch and the old model file and is left out.
  python3 ready_proc_k.py [--extra-real]   (--extra-real adds the E3-B tasks, see ../e3_more_tasks)
-> out[_extended].txt, results[_extended].json"""
import sys, os, json, importlib.util, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; SCR = HERE.parent
spec = importlib.util.spec_from_file_location("rp", SCR / "e5_embed" / "ready_proc_e5.py"); rp = importlib.util.module_from_spec(spec)
sys.argv_saved = sys.argv; spec.loader.exec_module(rp)
ap = argparse.ArgumentParser(); ap.add_argument("--extra-real", action="store_true"); A = ap.parse_args()
s1 = json.load(open(SCR / "tib" / "selection.json")); s2 = json.load(open(SCR / "combo_rep" / "selection_rep.json"))
F = s1["test_families"] + s2["test_families"]
z = np.load(SCR / "e5_embed" / "emb_tib.npz", allow_pickle=True); assert list(z["families"]) == F
SIM = z["X"] @ z["X"].T
GROUPS = {"gemini": (["g38", "flash"], {"real": ["tib/runs/real_{a}.jsonl", "combo_rep/runs/real_{a}.jsonl"],
                                         "k0": ["tib/runs/probes_{a}.v2.jsonl", "combo_rep/runs/doc_{a}.jsonl"],
                                         "k2": ["tib/runs/seeded_{a}.jsonl", "combo_rep/runs/seeded_{a}.jsonl"],
                                         "k5": ["seedk/runs/k5_{a}.jsonl"], "foreign": ["seedk/runs/foreign_{a}.jsonl"]}),
          "claude": (["haiku", "sonnet"], {"real": ["claude_tib/runs/real_{a}.jsonl"], "k0": ["claude_tib/runs/k0_{a}.jsonl"],
                                           "k2": ["claude_tib/runs/k2_{a}.jsonl"], "k5": ["claude_tib/runs/k5_{a}.jsonl"],
                                           "foreign": ["claude_tib/runs/foreign_{a}.jsonl"]})}
CONDS = ["k0", "k2", "k5", "foreign"]
EXTRA = {"g38": None, "flash": None, "haiku": SCR / "claude_tib/runs/new_haiku.jsonl", "sonnet": SCR / "claude_tib/runs/new_sonnet.jsonl"}
SEEDS = set()
for f, v in json.load(open(SCR / "seedk" / "selection_seedk.json"))["families"].items():
    SEEDS |= set(v.get("seeds_k1", []) + v.get("seeds_k2", []) + v.get("seeds_k5", []))


def load(paths, a, extra=None):
    s, n, seen = np.zeros(len(F)), np.zeros(len(F)), set()
    for p in [SCR / q.format(a=a) for q in paths] + ([extra] if extra else []):
        if not Path(p).exists(): continue
        for l in open(p):
            r = json.loads(l)
            if r.get("reward") is None or r["task_id"] in seen: continue
            if extra is not None and Path(p) == Path(extra) and r["task_id"] in SEEDS: continue
            seen.add(r["task_id"]); j = next(i for i, f in enumerate(F) if f"_{f}_" in r["task_id"])
            n[j] += 1; s[j] += int(r["reward"] >= 1.0)
    return s, n


RES = {}; rng = np.random.default_rng(0)
print("similarity: bge (no family prefix); extended real tasks:", A.extra_real)
for g, (AG, files) in GROUPS.items():
    for a in AG:
        if A.extra_real and EXTRA[a] is None: continue
        sR, nR = load(files["real"], a, EXTRA[a] if A.extra_real else None); rate = sR / nR
        probes = {c: load(files[c], a) for c in CONDS}
        print(f"\n{g}/{a}: real success {sR.sum() / nR.sum():.3f} on {int(nR.sum())} cases; probes per condition " + ", ".join(f"{c} {int(probes[c][1].sum())}" for c in CONDS))
        names = ["const (0 probes)", "knn text (0 probes)"] + [f"{c} {e}" for c in CONDS for e in ("raw", "bb", "bb_knn")]
        acc = {e: {"brier": [], **{f"{t:.2f}": [] for t in rp.TARGETS}} for e in names}
        for sp in range(1000):
            perm = rng.permutation(len(F)); D, Q = perm[:25], perm[25:]; pi = sR[D].sum() / nR[D].sum(); kn = np.zeros(len(F))
            for j in Q: kn[j] = rp.knn(SIM[j, D], rate[D])
            for j in D:
                o = D[D != j]; kn[j] = rp.knn(SIM[j, o], rate[o])
            est = {"const (0 probes)": np.full(len(F), pi), "knn text (0 probes)": kn}
            for c, (k, n) in probes.items():
                ok = n > 0; kk = np.where(ok, k, 0); nn = np.where(ok, n, 0)
                est[f"{c} raw"] = np.where(ok, kk / np.maximum(nn, 1), pi)
                for nm, prior in (("bb", np.full(len(F), pi)), ("bb_knn", kn)):
                    mb = min(rp.MS, key=lambda mm: rp.brier(((mm * prior + kk) / (mm + nn))[D], sR[D], nR[D]))
                    est[f"{c} {nm}"] = (mb * prior + kk) / (mb + nn)
            for e in names:
                acc[e]["brier"].append(rp.brier(est[e][Q], sR[Q], nR[Q]))
                for t in rp.TARGETS:
                    acc[e][f"{t:.2f}"].append(rp.qualify(est[e][D], sR[D], nR[D], est[e][Q], sR[Q], nR[Q], t, rng))
        res = {e: {"brier": float(np.mean(acc[e]["brier"])), **{f"{t:.2f}": rp.agg(acc[e][f"{t:.2f}"]) for t in rp.TARGETS}} for e in names}
        rp.show(f"TIB / {g}/{a}  (always-agent success {sR.sum() / nR.sum():.3f})", res, names)
        RES[f"{g}/{a}"] = res
suffix = "_extended" if A.extra_real else ""
json.dump(RES, open(HERE / f"results{suffix}.json", "w"), indent=1, default=float)
