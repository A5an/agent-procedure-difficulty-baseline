"""E8 / E8-0 analysis (PREREG.md). Re-scores every episode with the SOPBench evaluator, then the paired tests."""
import os, sys, json, csv, glob, collections, importlib.util
import numpy as np
from math import comb
E8 = os.path.dirname(os.path.abspath(__file__))
H = os.path.join(os.path.dirname(E8), "ideas2", "harness")
spec = importlib.util.spec_from_file_location("an", H + "/analyze.py"); AN = importlib.util.module_from_spec(spec)
sys.argv = sys.argv[:1]; spec.loader.exec_module(AN)          # loads tasks, evaluator, flags(), public()
tasks = AN.tasks
rows = list(csv.DictReader(open(H + "/sample.csv")))
SAMPLE = [r["case_id"] for r in rows]; PROC = {r["case_id"]: r["domain"] + "/" + r["procedure"] for r in rows}
PERFORM = [c for c in SAMPLE if tasks[c]["action_should_succeed"]]
W_PERFORM = 288 / 830


def load(fname):
    d = {}
    p = os.path.join(E8, "episodes", fname)
    if not os.path.exists(p): return d
    for l in open(p):
        r = json.loads(l)
        if r.get("error") or "interaction" not in r: continue
        if r["case_id"] in d: continue      # two runners overlapped for a few minutes on 13 Oct: keep the first record
        f = AN.episode_metrics(r, strip=False)
        f["router"] = r.get("router")
        f["temp_retry"] = r.get("temp", 0) > 0
        f["usd_by_model"] = collections.Counter()
        for u in r["usage"]: f["usd_by_model"][u.get("model", "single")] += u["usd"]
        d[r["case_id"]] = f
    return d


def summ(d, ids):
    ids = [c for c in ids if c in d]
    if not ids: return None
    s = AN.summarize(d, ids)
    s["success_reweighted"] = W_PERFORM * s["success_perform"] + (1 - W_PERFORM) * s["success_refuse"]
    s["temp_retry"] = float(np.mean([d[c].get("temp_retry", 0) for c in ids]))
    s["action_rate_perform"] = float(np.mean([d[c]["performed"] for c in ids if d[c]["should"]])) if any(d[c]["should"] for c in ids) else None
    s["empty_msg_share"] = float(sum(d[c].get("empty_msgs", 0) for c in ids) / max(1, sum(d[c].get("asst_msgs", 0) for c in ids)))
    s["trunc_calls"] = float(sum(d[c].get("trunc", 0) for c in ids) / max(1, sum(d[c]["calls"] for c in ids)))
    return s


def paired(a, b, ids, key="success", B=10000, seed=1):
    ids = [c for c in ids if c in a and c in b]
    if not ids: return {"n": 0}
    x = np.array([float(a[c][key]) for c in ids]); y = np.array([float(b[c][key]) for c in ids])
    n10, n01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    n = n10 + n01   # exact two-sided McNemar (sign test on the discordant pairs)
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(n10, n01) + 1)) / 2 ** n) if n else 1.0
    procs = sorted({PROC[c] for c in ids}); by = collections.defaultdict(list)
    for i, c in enumerate(ids): by[PROC[c]].append(i)
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(B):
        pick = rng.integers(0, len(procs), len(procs))
        ix = np.concatenate([by[procs[j]] for j in pick])
        diffs.append((x[ix] - y[ix]).mean())
    return {"n": len(ids), "diff": float((x - y).mean()), "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5)),
            "a_only": n10, "b_only": n01, "mcnemar_p": float(p)}


def handoff(d, alone, ids):
    ids = [c for c in ids if c in d and d[c]["router"]]
    h = [c for c in ids if d[c]["router"]["log"]]
    same_call = [c for c in h if d[c]["router"]["log"][0]["b_first"].get("call") == d[c]["router"]["log"][0]["a_proposal"]["call"]]
    same_args = [c for c in same_call if json.loads(d[c]["router"]["log"][0]["b_first"]["args"] or "{}") == json.loads(d[c]["router"]["log"][0]["a_proposal"]["args"] or "{}")]
    out = {"episodes": len(ids), "handoff_share": len(h) / max(1, len(ids)),
           "handoff_perform": sum(tasks[c]["action_should_succeed"] for c in h), "handoff_refuse": sum(not tasks[c]["action_should_succeed"] for c in h),
           "b_repeats_target_call": len(same_call) / max(1, len(h)), "b_repeats_same_args": len(same_args) / max(1, len(h)),
           "b_first_other": collections.Counter((d[c]["router"]["log"][0]["b_first"].get("call") or "text") for c in h if c not in same_call).most_common(8),
           "success_in_handoff": float(np.mean([d[c]["success"] for c in h])) if h else None}
    if alone:
        hh = [c for c in h if c in alone]
        out["a_alone_success_same_cases"] = float(np.mean([alone[c]["success"] for c in hh])) if hh else None
        nh = [c for c in ids if c not in h and c in alone]
        out["no_handoff_agree_with_a_alone"] = float(np.mean([d[c]["success"] == alone[c]["success"] for c in nh])) if nh else None
    return out


if __name__ == "__main__":
    files = {"flash": "flash.jsonl", "pro": "pro.jsonl", "pro>flash": "pro_to_flash.jsonl", "flash>pro": "flash_to_pro.jsonl",
             "lite": "lite.jsonl", "g38": "g38.jsonl", "lite>g38": "lite_to_g38.jsonl", "pro@512 (E8-0)": "pro__e80.jsonl"}
    D = {k: load(v) for k, v in files.items()}
    D["public flash (512)"] = AN.public("gemini-2.5-flash"); D["public pro (512)"] = AN.public("gemini-2.5-pro")
    D["harness study flash (5 Oct)"] = AN.load_arm("gemini-2.5-flash", "H0"); D["harness study g38 (5 Oct)"] = AN.load_arm("gemini-3.8-flash", "H0")
    res = {"arms": {}, "paired": {}, "handoff": {}, "e80": {}}
    for k, d in D.items():
        if d: res["arms"][k] = summ(d, SAMPLE)
    # primary contrast, fixed in PREREG: pro>flash minus the better single of flash and pro (raw success on the 150)
    if all(D[k] for k in ("flash", "pro", "pro>flash")):
        best = max(("flash", "pro"), key=lambda k: res["arms"][k]["success"])
        res["primary"] = {"best_single": best, **paired(D["pro>flash"], D[best], SAMPLE)}
        for key in ("success", "unsafe", "wrongful"):
            for a, b in (("pro>flash", "pro"), ("pro>flash", "flash"), ("flash>pro", "flash"), ("flash>pro", "pro"), ("pro", "flash")):
                if D[a] and D[b]: res["paired"][f"{a} - {b} [{key}]"] = paired(D[a], D[b], SAMPLE, key)
        for a, b in (("pro>flash", "pro"), ("pro>flash", "flash")):
            for sub, ids in (("perform", PERFORM), ("refuse", [c for c in SAMPLE if c not in PERFORM])):
                res["paired"][f"{a} - {b} [success, {sub}]"] = paired(D[a], D[b], ids)
    for a, b in (("lite>g38", "g38"), ("lite>g38", "lite"), ("g38", "lite"), ("flash", "harness study flash (5 Oct)"), ("g38", "harness study g38 (5 Oct)")):
        if D[a] and D[b]: res["paired"][f"{a} - {b} [success]"] = paired(D[a], D[b], SAMPLE)
    for m, alone in (("pro>flash", "pro"), ("flash>pro", "flash"), ("lite>g38", "lite")):
        if D[m]: res["handoff"][m] = handoff(D[m], D[alone], SAMPLE)
    # E8-0: pro at 512 vs 4096 on the perform cases
    if D["pro@512 (E8-0)"] and D["pro"]:
        for k in ("pro@512 (E8-0)", "pro", "public pro (512)"):
            res["e80"][k] = summ(D[k], PERFORM)
        res["e80"]["4096 - 512 [success]"] = paired(D["pro"], D["pro@512 (E8-0)"], PERFORM)
        res["e80"]["4096 - 512 [performed]"] = paired(D["pro"], D["pro@512 (E8-0)"], PERFORM, "performed")
    json.dump(res, open(os.path.join(E8, "results.json"), "w"), indent=1, default=str)
    cols = ["n", "success", "success_reweighted", "success_perform", "success_refuse", "unsafe_refuse", "wrongful_perform", "dirgraph",
            "empty_ep", "trunc", "calls", "think", "usd_total", "temp_retry"]
    print("arm".ljust(30), " ".join(c[:9].rjust(9) for c in cols))
    for k, s in res["arms"].items():
        print(k.ljust(30), " ".join((f"{s[c]:9.3f}" if isinstance(s[c], float) else f"{s[c]:9d}") for c in cols))
    if "primary" in res: print("\nPRIMARY", res["primary"])
    for k, v in res["paired"].items(): print(k.ljust(48), {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items()})
    for k, v in res["handoff"].items(): print("\nHANDOFF", k, v)
    for k, v in res["e80"].items(): print("\nE8-0", k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items()} if isinstance(v, dict) else v)
