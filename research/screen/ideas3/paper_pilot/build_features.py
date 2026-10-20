"""Plans -> features/pf_<model>_<bench>.csv, pf_both_<bench>.csv, plans_parsed.json"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re, sys, os
import numpy as np, pandas as pd
import pp_lib as L
HERE = L.HERE; REPO = L.REPO
COMP = ZR.SCREEN + "/ideas2/compile/"
MODELS = {"lite": "gemini-2.5-flash-lite", "g38": "gemini-3.8-flash", "f25": "gemini-2.5-flash"}
os.makedirs(HERE + "/features", exist_ok=True)

def required_tools(code, goal):   # same rule as ideas2/harness/run.py
    pos = [m.start() for m in re.finditer(r"tools\.%s\b" % re.escape(goal), code)]
    cut = pos[0] if pos else len(code)
    names = []
    for m in re.finditer(r"tools\.(\w+)", code[:cut]):
        if m.group(1) not in names and m.group(1) != goal: names.append(m.group(1))
    return names

def parse(text):
    if not text: return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    try: d = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if not m: return None
        try: d = json.loads(m.group(0))
        except Exception: return None
    if not isinstance(d, dict): return None
    calls = []
    for c in d.get("tool_calls") or []:
        if isinstance(c, dict) and c.get("name"): calls.append(str(c["name"]))
    dec = str(d.get("final_decision", "")).lower().strip()
    rc = d.get("refusal_conditions") or []
    return {"calls": calls, "dec": dec, "rc": len(rc) if isinstance(rc, list) else 1}

def stmts(bench): return {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(f"{REPO}/data/{bench}/statements.jsonl")}

def sop_meta():
    units = {r["case_id"]: r["unit"] for r in json.load(open(COMP + "results_A.json"))}
    progs = json.load(open(COMP + "programs_t0.json"))
    return {cid: required_tools(progs[u]["code"], cid.split("/")[1]) for cid, u in units.items()}

WRITE = re.compile(r"^(book|cancel|update|modify|exchange|return|send|enable|disable|suspend|resume|refuel)")
def common(p):
    d = p["dec"]; calls = p["calls"]
    return {"n_calls": len(calls), "n_unique": len(set(calls)), "dec_perform": float(d == "perform"), "dec_refuse": float(d == "refuse"),
            "dec_transfer": float(d == "transfer"), "refusal_path": float(p["rc"] > 0 or d == "refuse"), "n_refusal_conds": p["rc"]}

def sop_feats(p, goal, req):
    f = common(p); calls = p["calls"]
    ia = calls.index(goal) if goal in calls else None
    first = {}
    for i, c in enumerate(calls): first.setdefault(c, i)
    miss = [r for r in req if r not in first]
    after = [r for r in req if r in first and ia is not None and first[r] > ia]
    extra = [c for c in set(calls) if c != goal and c not in req and c != "exit_conversation"]
    n = max(len(req), 1)
    f.update(calls_action=float(ia is not None), n_req=len(req), n_missing=len(miss), frac_missing=len(miss) / n,
             n_after=len(after), frac_after=len(after) / n, frac_bad=(len(miss) + len(after)) / n,
             action_first=float(ia == 0), n_before_action=float(ia if ia is not None else len(calls)), n_extra=len(extra),
             any_bad=float(len(miss) + len(after) > 0))
    return f

def tau_feats(p):
    f = common(p); calls = p["calls"]
    w = [i for i, c in enumerate(calls) if WRITE.match(c)]; r = [i for i, c in enumerate(calls) if c.startswith(("get_", "find_", "search_", "list_", "check_", "calculate"))]
    tr = [i for i, c in enumerate(calls) if c == "transfer_to_human_agents"]
    f.update(n_write=len(w), n_read=len(r), n_transfer_call=len(tr),
             write_before_read=float(bool(w) and (not r or w[0] < r[0])),
             reads_before_write=float(sum(1 for i in r if w and i < w[0]) if w else len(r)),
             write_no_read=float(bool(w) and not r))
    return f

def build():
    meta = sop_meta(); rep = {}; parsed = {}
    for bench in ("sopbench", "tau2"):
        S = stmts(bench); ids = list(S); per = {}
        for mk, mname in MODELS.items():
            cache = L.load_cache(mname)
            rows = {}; nf = 0
            for cid in ids:
                r = cache.get(L.key(mname, L.build_prompt(cid, S[cid])))
                p = parse(r["text"]) if r else None
                parsed.setdefault(mk, {})[cid] = p
                if p is None: nf += 1; continue
                rows[cid] = sop_feats(p, cid.split("/")[1], meta[cid]) if bench == "sopbench" else tau_feats(p)
            df = pd.DataFrame.from_dict(rows, orient="index")
            rep[f"{bench}|{mk}"] = {"n": len(ids), "missing": nf}
            if len(df) == 0: continue
            df = df.reindex(ids); df = df.fillna(df.median()); df.index.name = "case_id"; per[mk] = df
            df.reset_index().to_csv(f"{HERE}/features/pf_{mk}_{bench}.csv", index=False)
        if "lite" in per and "g38" in per:   # both models + disagreement
            a, b = per["lite"], per["g38"]; dis = pd.DataFrame(index=a.index)
            pl, pg = parsed["lite"], parsed["g38"]
            def jac(i):
                x, y = pl[i], pg[i]
                if x is None or y is None: return np.nan
                s, t = set(x["calls"]), set(y["calls"]); return 1 - len(s & t) / max(len(s | t), 1)
            dis["dec_disagree"] = [float(pl[i] is not None and pg[i] is not None and pl[i]["dec"] != pg[i]["dec"]) if pl[i] and pg[i] else np.nan for i in a.index]
            dis["tool_jaccard_dist"] = [jac(i) for i in a.index]
            dis["abs_ncalls_diff"] = (a.n_calls - b.n_calls).abs()
            dis = dis.fillna(dis.median())
            both = pd.concat([a.add_prefix("l_"), b.add_prefix("g_"), dis], axis=1); both.index.name = "case_id"
            both.reset_index().to_csv(f"{HERE}/features/pf_both_{bench}.csv", index=False)
    json.dump(parsed, open(HERE + "/plans_parsed.json", "w"))
    json.dump(rep, open(HERE + "/parse_report.json", "w"), indent=1); print(json.dumps(rep))

if __name__ == "__main__": build()
