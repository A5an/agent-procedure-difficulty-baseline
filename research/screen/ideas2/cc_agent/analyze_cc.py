import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os, sys, json, csv, glob, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score as S
ids = S.ids
SOPORIG = S.SOP + "/output/"
COMP = ZR.SCREEN + "/ideas2/compile/"
REPO = ZR.REPO + "/data/sopbench/responses.jsonl"
flags = S.flags
CARRY = {r['case_id']: r['carry'] == 'True' for r in csv.DictReader(open(S.SAMPLE))}
def public(subject):
    out = {}
    for d in ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]:
        fn = glob.glob(f"{SOPORIG}{d}/ast_{subject}-dep_full-fmt_structured-tool_full-shuffle_False.json")
        if not fn: continue
        cnt = collections.Counter()
        for r in json.load(open(fn[0])):
            g = r["task"]["user_goal"]; cid = f"{d}/{g}/{cnt[g]}"; cnt[g] += 1
            ev = r["evaluations"][0] if r.get("evaluations") else None
            if ev is None: continue
            f = flags(ev, bool(r["task"]["action_should_succeed"]))
            inter = r["interactions"][0]["interaction"]
            f["calls"] = sum(len(m["tool_calls"]) for m in inter if m.get("tool_calls") and m.get("sender", "") not in ("user", ""))
            f["usd"] = 0; f["blocks"] = 0
            out[cid] = f
    return out
def compiled(fn):
    d = {}
    for r in json.load(open(COMP + fn)):
        should = bool(r["should"]); done = bool(r["action_successfully_called"])
        d[r["case_id"]] = dict(success=bool(r["success"]), decision=(done == should), should=should, unsafe=(not should) and (done or not r["database_match"]),
                               wrongful=should and not done, performed=done, calls=r["ncalls"], usd=0, blocks=0)
    return d
def summ(d):
    c = [x for x in ids if x in d]
    g = lambda k, s=None: [float(d[x][k]) for x in c if k in d[x] and (s is None or d[x]["should"] == s)]
    m = lambda v: float(np.mean(v)) if v else float("nan")
    return dict(n=len(c), success=m(g("success")), perform=m(g("success", True)), refuse=m(g("success", False)), decision=m(g("decision")),
                unsafe=m(g("unsafe", False)), wrongful=m(g("wrongful", True)), calls=m(g("calls")), blocks=m(g("blocks")), usd=float(np.sum(g("usd"))), usd_ep=m(g("usd")))
def boot(a, b, key="success", B=10000, seed=1):
    c = [x for x in ids if x in a and x in b and key in a[x] and key in b[x]]
    x = np.array([float(a[i][key]) for i in c]); y = np.array([float(b[i][key]) for i in c])
    rng = np.random.default_rng(seed); n = len(c); idx = rng.integers(0, n, (B, n)); df = (x[idx] - y[idx]).mean(1)
    return dict(n=n, diff=float((x - y).mean()), lo=float(np.percentile(df, 2.5)), hi=float(np.percentile(df, 97.5)), a_only=int(((x==1)&(y==0)).sum()), b_only=int(((x==0)&(y==1)).sum()))
if __name__ == "__main__":
    # public subjects on the 150 cases
    subj = {}
    for l in open(REPO):
        r = json.loads(l); resp = r["responses"]
        v = [resp[c] for c in ids if c in resp]
        subj[r["subject_id"]] = (np.mean(v), len(v))
    rank = sorted(subj, key=lambda s: subj[s][0])
    med = rank[len(rank)//2 - 1]  # 28 subjects: the 14th of 28 from the bottom (lower median)
    print("n subjects", len(rank), "median subject", med, subj[med], "best", rank[-1], subj[rank[-1]], "second", rank[-2], subj[rank[-2]])
    print("o4-mini-high subject ids:", [s for s in rank if "o4-mini-high" in s])
    D = {}
    for arm in ["native", "guard", "native_date", "guard_date"]:
        if os.path.exists(f"{HERE}/out/{arm}"): D[f"CC Sonnet {arm}"] = S.load_arm(arm)
    ob = [s for s in rank if s.startswith("o4-mini-high")]
    D["public o4-mini-high"] = public("o4-mini-high-mode_fc"); D[f"public median ({med})"] = public(med)
    # median agent: success from the repo responses for all 150 cases, flags from the logs where they exist
    resp = {json.loads(l)["subject_id"]: json.loads(l)["responses"] for l in open(REPO)}
    k = f"public median ({med})"; full = {}
    for c in ids:
        base = dict(D[k].get(c, {})); base["success"] = bool(resp[med][c]); full[c] = base
    for c in ids:
        full[c]["should"] = CARRY[c]
    D[k] = {c: v for c, v in full.items()}
    hc = os.path.join(HERE, "harness_cases.json")
    if os.path.exists(hc):
        for kk, v in json.load(open(hc)).items(): D["harness " + kk] = v
    D["compiled A"] = compiled("results_A.json"); D["compiled AC"] = compiled("results_AC.json")
    for f in glob.glob(f"{HERE}/../harness/episodes/*.jsonl"): pass
    H = glob.glob(f"{HERE}/../harness/results.json")
    print(f"{'arm':40s} n  succ  perf  refu  dec  unsafe wrong calls blocks $tot $/ep")
    res = {"arms": {k: summ(v) for k, v in D.items()}, "paired": {}}
    for k, s in res["arms"].items(): print(f"{k:40s} {s['n']:3d} {s['success']:.3f} {s['perform']:.3f} {s['refuse']:.3f} {s['decision']:.3f} {s['unsafe']:.3f} {s['wrongful']:.3f} {s['calls']:.2f} {s['blocks']:.2f} {s['usd']:.2f} {s['usd_ep']:.3f}")
    for cc in [k for k in D if k.startswith("CC")]:
        for o in [k for k in D if not k.startswith("CC")]:
            for key in ("success", "unsafe", "wrongful"):
                res["paired"][f"{cc} - {o} [{key}]"] = boot(D[cc], D[o], key)
        for key in ("success", "unsafe", "wrongful"):
            if "CC Sonnet native" in D and cc != "CC Sonnet native": res["paired"][f"{cc} - CC Sonnet native [{key}]"] = boot(D[cc], D["CC Sonnet native"], key)
    for k, v in res["paired"].items(): print(f"{k:70s} {v['diff']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] n={v['n']} a_only={v['a_only']} b_only={v['b_only']}")
    json.dump(res, open(f"{HERE}/results.json", "w"), indent=1)
