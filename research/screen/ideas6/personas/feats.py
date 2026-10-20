"""Persona plans -> features/pers_<bench>.csv (and stability features from temp 0.7 run)."""
import json, re, sys, os
import numpy as np, pandas as pd
import plib as L
sys.path.insert(0, L.PP)
import build_features as BF          # required_tools rule, sop_meta
HERE = L.HERE; os.makedirs(HERE + "/features", exist_ok=True)
TT = json.load(open(L.PT + "/tool_types.json"))
def parse(text):
    if not text: return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try: d = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if not m: return None
        try: d = json.loads(m.group(0))
        except Exception: return None
    ps = d.get("personas") if isinstance(d, dict) else None
    if not isinstance(ps, list) or len(ps) != 5: return None
    out = []
    for p in ps:
        if not isinstance(p, dict): return None
        calls = [str(c) for c in (p.get("tool_calls") or []) if isinstance(c, (str,))]
        sk = p.get("skipped_conditions") or []
        out.append({"calls": calls, "dec": str(p.get("final_decision", "")).lower().strip(), "skip": len(sk) if isinstance(sk, list) else 1})
    return out
def agg(rows):
    """rows: list of 5 dicts with ok, awc, anone, dec, skip, calls, bad"""
    ok = np.array([r["ok"] for r in rows], float); decs = [r["dec"] for r in rows]
    top = max(decs.count(d) for d in set(decs))
    idx = [i for i, o in enumerate(ok) if o]
    return {"share_ok": ok.mean(), "n_act_wo_check": sum(r["awc"] for r in rows), "n_act_nocheck": sum(r["anone"] for r in rows),
            "dec_disagree": 1 - top / 5, "share_perform": decs.count("perform") / 5, "share_refuse": decs.count("refuse") / 5,
            "weakest_ok": (idx[0] + 1) if idx else 6, "none_ok": float(not idx), "mean_bad": np.mean([r["bad"] for r in rows]),
            "skip_mean": np.mean([r["skip"] for r in rows]), "skip_gap": rows[0]["skip"] - rows[4]["skip"],
            "calls_mean": np.mean([len(r["calls"]) for r in rows]), "calls_gap": len(rows[4]["calls"]) - len(rows[0]["calls"])}
def sop_rows(ps, goal, req):
    rows = []
    for p in ps:
        calls = p["calls"]; first = {}
        for i, c in enumerate(calls): first.setdefault(c, i)
        ia = first.get(goal)
        miss = [r for r in req if r not in first]; after = [r for r in req if r in first and ia is not None and first[r] > ia]
        n = max(len(req), 1); nbad = len(miss) + len(after)
        rows.append({"ok": float(nbad == 0), "awc": float(ia is not None and nbad > 0), "anone": float(ia is not None and len(req) > 0 and len(miss) + len(after) == len(req)),
                     "dec": p["dec"], "skip": p["skip"], "calls": calls, "bad": nbad / n})
    return rows
def tau_rows(ps, dom, plan):
    Rr, Rw = round(plan["plan_reads"]), round(plan["plan_writes"]); rows = []
    for p in ps:
        calls = [c for c in p["calls"] if f"{dom}/{c}" in TT]
        ty = [TT[f"{dom}/{c}"] for c in calls]
        r = [i for i, t in enumerate(ty) if t == "read"]; w = [i for i, t in enumerate(ty) if t == "write"]
        wnr = bool(w) and (not r or w[0] < r[0])
        short_r = max(Rr - len(r), 0); short_w = max(Rw - len(w), 0)
        bad = (short_r + short_w + (1 if wnr else 0)) / max(Rr + Rw, 1)
        rows.append({"ok": float(short_r == 0 and short_w == 0 and not wnr), "awc": float(bool(w) and (wnr or short_r > 0)), "anone": float(wnr),
                     "dec": p["dec"], "skip": p["skip"], "calls": calls, "bad": bad})
    return rows
def build(temp=0.0,subset=None, tag="pers"):
    meta = BF.sop_meta(); cache = L.load_cache(); rep = {}; allf = {}
    for bench in ("sopbench", "tau2"):
        S = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(f"{L.REPO}/data/{bench}/statements.jsonl")}
        plan = pd.read_csv(L.PT + "/features/plan3_tau2.csv").set_index("case_id") if bench == "tau2" else None
        rows = {}; nf = 0
        for cid, st in S.items():
            if subset is not None and cid not in subset.get(bench, ()): continue
            r = cache.get(L.key(L.build_prompt(cid, st), temp)); ps = parse(r["text"]) if r else None
            if ps is None: nf += 1; continue
            rr = sop_rows(ps, cid.split("/")[1], meta[cid]) if bench == "sopbench" else tau_rows(ps, cid.split("/")[0], plan.loc[cid])
            rows[cid] = agg(rr)
        df = pd.DataFrame.from_dict(rows, orient="index"); rep[bench] = {"n": len(S) if subset is None else len(rows), "parse_fail": nf}
        if subset is None:
            df = df.reindex(list(S)); df = df.fillna(df.median())
        df.index.name = "case_id"; allf[bench] = df
        df.reset_index().to_csv(f"{HERE}/features/{tag}_{bench}.csv", index=False)
    print(json.dumps(rep)); return allf
if __name__ == "__main__": build()
