import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os, sys, json, csv, glob, copy, random, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/sop"); os.chdir(HERE + "/sop")
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("runmod", HERE + "/run.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
tasks = R.load_tasks()
SOPORIG = ZR.DATA + "/SOPBench-d2622008/output/"
COMP = R.COMP
sample = [r["case_id"] for r in csv.DictReader(open(HERE + "/sample.csv"))]
suffix = sys.argv[1] if len(sys.argv) > 1 else ""

def flags(ev, should):
    done = bool(ev["action_successfully_called"])
    return dict(success=bool(ev["success"]), decision=(done == should), should=should,
                unsafe=(not should) and (done or not ev["database_match"]),
                wrongful=should and not done, performed=done,
                dirgraph=bool(ev["dirgraph_satisfied"]), correct_call=bool(ev["action_called_correctly"]))

def episode_metrics(rec, strip):
    t = tasks[rec["case_id"]]; should = bool(t["action_should_succeed"])
    ev = R.evaluate(t, t["domain"], rec["interaction"], rec["database"], strip=strip)
    f = flags(ev, should)
    asst = [m for m in rec["interaction"] if m.get("sender") and m["sender"] not in ("user",) and m.get("role", "assistant") != "tool" and "tool_name" not in m]
    asst = [m for m in rec["interaction"] if m.get("sender", "") not in ("user", "") and "tool_name" not in m]
    empty = [m for m in asst if not (m.get("content") or "").strip() and not m.get("tool_calls")]
    f["empty_msgs"] = len(empty); f["asst_msgs"] = len(asst); f["has_empty"] = len(empty) > 0
    f["no_final_text"] = False
    f["calls"] = len(rec["usage"]); f["usd"] = sum(u["usd"] for u in rec["usage"])
    f["think"] = sum(u.get("think", 0) for u in rec["usage"]); f["intok"] = sum(u["in"] for u in rec["usage"])
    f["trunc"] = sum(1 for u in rec["usage"] if u.get("finish") in ("MAX_TOKENS", "max_tokens"))
    f["blocks"] = len(rec.get("blocks") or [])
    return f

def load_arm(model, arm, strip=False):
    p = f"{R.OUT}/{model}__{arm}{suffix}.jsonl"
    d = {}
    if not os.path.exists(p): return d
    for l in open(p):
        r = json.loads(l)
        if r.get("error") or "interaction" not in r: continue
        d[r["case_id"]] = episode_metrics(r, strip)
    return d

def public(agent):
    """public SOPBench log for agent on the sample (evaluations[0])"""
    out = {}
    for d in ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]:
        fn = glob.glob(f"{SOPORIG}{d}/ast_{agent}-mode_fc-dep_full-fmt_structured-tool_full-shuffle_False.json")
        if not fn: continue
        recs = json.load(open(fn[0])); cnt = collections.Counter()
        for r in recs:
            g = r["task"]["user_goal"]; cid = f"{d}/{g}/{cnt[g]}"; cnt[g] += 1
            ev = r["evaluations"][0] if r.get("evaluations") else None
            if ev is None: continue
            f = flags(ev, bool(r["task"]["action_should_succeed"]))
            inter = r["interactions"][0]["interaction"]
            asst = [m for m in inter if m.get("sender", "") not in ("user", "") and "tool_name" not in m]
            f["has_empty"] = any(not (m.get("content") or "").strip() and not m.get("tool_calls") for m in asst)
            f["calls"] = sum(1 for m in asst if m.get("tool_calls")) + sum(1 for m in asst if not m.get("tool_calls"))
            f['usd'] = 0
            out[cid] = f
    return out

def compiled(fn):
    d = {}
    for r in json.load(open(COMP + fn)):
        should = bool(r["should"]); done = bool(r["action_successfully_called"])
        d[r["case_id"]] = dict(success=bool(r["success"]), decision=(done == should), should=should, unsafe=(not should) and (done or not r["database_match"]),
                               wrongful=should and not done, performed=done, dirgraph=bool(r["dirgraph_satisfied"]), correct_call=bool(r["action_called_correctly"]), calls=r["ncalls"], has_empty=False, usd=0)
    return d

def summarize(d, ids=None):
    ids = [c for c in (ids or sample) if c in d]; n = len(ids)
    if n == 0: return None
    g = lambda k, sub=None: [d[c][k] for c in ids if sub is None or d[c]["should"] == sub]
    m = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return dict(n=n, success=m(g("success")), success_perform=m(g("success", True)), success_refuse=m(g("success", False)), decision=m(g("decision")),
                unsafe_refuse=m(g("unsafe", False)), wrongful_perform=m(g("wrongful", True)), empty_ep=m(g("has_empty")),
                calls=m(g("calls")), usd_per_ep=m(g("usd")), usd_total=float(np.sum(g("usd"))), think=m([d[c].get("think", 0) for c in ids]),
                dirgraph=m(g("dirgraph")), trunc=m([d[c].get("trunc", 0) > 0 for c in ids]), blocks=m([d[c].get("blocks", 0) > 0 for c in ids]))

# paired bootstrap, resampling cases
def boot(a, b, key="success", B=10000, seed=1, ids=None):
    ids = [c for c in (ids or sample) if c in a and c in b]
    x = np.array([float(a[c][key]) for c in ids]); y = np.array([float(b[c][key]) for c in ids])
    rng = np.random.default_rng(seed); n = len(ids)
    idx = rng.integers(0, n, (B, n)); diff = (x[idx] - y[idx]).mean(1)
    return dict(n=n, diff=float((x - y).mean()), lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)),
                a_only=int(((x == 1) & (y == 0)).sum()), b_only=int(((x == 0) & (y == 1)).sum()))

if __name__ == "__main__":
    models = [m for m in ["gemini-2.5-flash", "gemini-3.8-flash", "claude"] ]
    arms = {}
    for f in glob.glob(f"{R.OUT}/*__*{suffix}.jsonl"):
        b = os.path.basename(f)[:-6]; mo, ar = b.split("__")
        if suffix and not ar.endswith(suffix): continue
        if not suffix and "_" in ar: continue
        ar = ar[:len(ar) - len(suffix)] if suffix else ar
        arms[(mo, ar)] = None
    res = {"arms": {}, "paired": {}}
    D = {}
    for (mo, ar) in sorted(arms):
        D[(mo, ar)] = load_arm(mo, ar, strip=(ar == "H2"))
        if ar == "H2": D[(mo, "H2raw")] = load_arm(mo, ar, strip=False)
    pub = {"public gemini-2.5-flash (512 cap)": public("gemini-2.5-flash"), "public o4-mini-high": public("o4-mini-high"),
           "compiled program A": compiled("results_A.json"), "compiled program AC": compiled("results_AC.json")}
    for k, v in pub.items(): D[(k, "")] = v
    for k, d in D.items(): res["arms"][f"{k[0]} {k[1]}".strip()] = summarize(d)
    mods = sorted({k[0] for k in D if k[1] in ("H0", "H1", "H2")})
    for mo in mods:
        for ar in ("H1", "H2", "H2raw"):
            if (mo, ar) in D and (mo, "H0") in D:
                for key in ("success", "unsafe", "wrongful"):
                    res["paired"][f"{mo} {ar} - H0 [{key}]"] = boot(D[(mo, ar)], D[(mo, "H0")], key)
        for pk in ("compiled program A", "compiled program AC"):
            for ar in ("H0", "H1", "H2"):
                if (mo, ar) in D: res["paired"][f"{mo} {ar} - {pk} [success]"] = boot(D[(mo, ar)], D[(pk, "")], "success")
    if ("gemini-2.5-flash", "H0") in D:
        res["paired"]["gemini-2.5-flash H0(4096) - public log (512) [success]"] = boot(D[("gemini-2.5-flash", "H0")], D[("public gemini-2.5-flash (512 cap)", "")], "success")
        for key in ("unsafe", "wrongful"):
            res["paired"][f"gemini-2.5-flash H0(4096) - public log (512) [{key}]"] = boot(D[("gemini-2.5-flash", "H0")], D[("public gemini-2.5-flash (512 cap)", "")], key)
    if ("gemini-3.8-flash", "H0") in D:
        res["paired"]["gemini-3.8-flash H0 - public o4-mini-high [success]"] = boot(D[("gemini-3.8-flash", "H0")], D[("public o4-mini-high", "")], "success")
    json.dump(res, open(HERE + f"/results{suffix}.json", "w"), indent=1)
    print(f"{'arm':44s} n   succ  perf  refu  dec   unsafe wrongf empty calls  $/ep   $tot  think blocks")
    for k, v in res["arms"].items():
        if v: print(f"{k:44s} {v['n']:3d} {v['success']:.3f} {v['success_perform']:.3f} {v['success_refuse']:.3f} {v['decision']:.3f} {v['unsafe_refuse']:.3f}  {v['wrongful_perform']:.3f}  {v['empty_ep']:.3f} {v['calls']:.1f} {v['usd_per_ep']:.4f} {v['usd_total']:.2f} {v['think']:.0f} {v['blocks']:.2f}")
    for k, v in res["paired"].items(): print(f"{k:70s} diff {v['diff']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] n={v['n']} a_only={v['a_only']} b_only={v['b_only']}")
