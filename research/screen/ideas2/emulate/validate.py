"""Step 3: validate every synthetic case with code. Ground truth label comes from the env's own check functions."""
import sys, os, json, copy, inspect, re, collections
import gt, gen
HERE = os.path.dirname(os.path.abspath(__file__))

def recur_data_consistency(d1, d2)->bool:
    """verbatim copy of env/check_data_sanity.recur_data_consistency (that module imports openai, which is not installed here)"""
    if not d1 or not d2 or not isinstance(d1, dict) and not isinstance(d2, dict):
        return True
    res = True
    d1_keys = d1.keys(); d2_keys = d2.keys()
    for d1_key in d1_keys:
        if d1_keys != d2_keys:
            for d2_key in d2_keys:
                if ((isinstance(d1[d1_key], dict) and isinstance(d2[d2_key], dict) and d1[d1_key].keys() != d2[d2_key].keys())
                    or (isinstance(d1[d1_key], dict) or isinstance(d2[d2_key], dict)) and type(d1[d1_key]) != type(d2[d2_key])):
                    return False
                res = res and recur_data_consistency(d1[d1_key], d2[d2_key])
        else:
            if (isinstance(d1[d1_key], dict) or isinstance(d2[d1_key], dict)) and type(d1[d1_key]) != type(d2[d1_key]):
                return True
            res = res and recur_data_consistency(d1[d1_key], d2[d1_key])
    return res

def goal_params(dom, proc, db, cp):
    dep_innate, default_dep, _ = gt.defaults(dom)
    s = gt.domain_keys[dom](copy.deepcopy(db), dep_innate, cp)
    sig = inspect.signature(getattr(s, proc))
    return s, [p for p, v in sig.parameters.items() if v.default is inspect.Parameter.empty and v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD)]

def match(ref, got, path="$"):
    """Lenient structural match of a generated value against the reference skeleton. Returns None or the first mismatch path."""
    if isinstance(ref, str):
        if ref == "null" or got is None: return None
        ok = {"str": isinstance(got, str), "bool": isinstance(got, bool),
              "int": isinstance(got, (int, float)) and not isinstance(got, bool),
              "float": isinstance(got, (int, float)) and not isinstance(got, bool)}.get(ref.split(" ")[0], True)
        return None if ok else path
    if isinstance(ref, list):
        if not isinstance(got, list): return path
        if ref and got:
            for j, x in enumerate(got):
                m = match(ref[0], x, f"{path}[{j}]")
                if m: return m
        return None
    if isinstance(ref, dict):
        if not isinstance(got, dict): return path
        if list(ref.keys()) == ["<key>"]:
            for k, x in got.items():
                m = match(ref["<key>"], x, f"{path}.{k}")
                if m: return m
            return None
        if set(ref.keys()) != set(got.keys()): return path + " keys"
        for k in ref:
            m = match(ref[k], got[k], f"{path}.{k}")
            if m: return m
    return None

def fail_pattern(res):
    return tuple(sorted(k for k, v in res.items() if not v))

def mentions(uk, msg):
    m = msg.lower().replace(",", "")
    miss = []
    for k, v in uk.items():
        sv = str(v).lower().replace(",", "")
        alts = {sv}
        if isinstance(v, float) and v == int(v): alts.add(str(int(v)))
        if isinstance(v, (int, float)): alts.add(f"{v:g}")
        if not any(a in m for a in alts): miss.append(k)
    return miss

def run():
    gens = json.load(open(os.path.join(HERE, sys.argv[1] if len(sys.argv)>1 else "gen_full.json")))
    rows = []
    for g in gens:
        dom, proc = g["dom"], g["proc"]
        cs, same = gen.modal_variant(dom, proc)
        ref = same[0]
        try:
            cases = json.loads(g["text"]); assert isinstance(cases, list)
        except Exception as e:
            rows.append({**{k: g[k] for k in ("dom","proc","variant","rep")}, "idx": -1, "parse_ok": False}); continue
        for i, c in enumerate(cases):
            r = {"dom": dom, "proc": proc, "variant": g["variant"], "rep": g["rep"], "idx": i, "parse_ok": True}
            try:
                db = c["initial_database"]; uk = c["user_known"]; msg = c["user_prompt"]; llm_lab = bool(c["should_perform"])
                r["keys_ok"] = isinstance(db, dict) and isinstance(uk, dict) and isinstance(msg, str)
                r["llm_label"] = llm_lab
            except Exception:
                r["keys_ok"] = False; rows.append(r); continue
            # structure: top level keys of the database equal the reference, and user_known keys equal
            r["top_keys_ok"] = set(db.keys()) == set(ref["initial_database"].keys())
            r["uk_keys_ok"] = set(uk.keys()) == set(ref["user_known"].keys())
            # skeleton equality against the reference schema
            r["schema_mismatch"] = match(gen.skeleton(ref["initial_database"]), db); r["schema_ok"] = r["schema_mismatch"] is None
            try:
                r["env_consistent"] = bool(recur_data_consistency(db, ref["initial_database"]) and recur_data_consistency(ref["initial_database"], db))
            except Exception as e:
                r["env_consistent"] = None
            case = copy.deepcopy(ref)
            case["initial_database"] = db; case["user_known"] = uk; case["user_prompt"] = msg
            ok, err = gt.load_check(dom, case, goal=proc)
            r["loads"] = ok; r["load_err"] = err
            # tools run: call the goal action on a fresh env with the user_known params
            try:
                dep_innate, default_dep, _ = gt.defaults(dom)
                dep_full = copy.deepcopy(default_dep); dep_full[proc] = case["constraints"]
                st = gt.domain_keys[dom + "_strict"](copy.deepcopy(db), dep_innate, dep_full, case["constraint_parameters"])
                params = [p for p, v in inspect.signature(getattr(st.domain_system if hasattr(st,'domain_system') else st.evaluation_get_domain_system(), proc)).parameters.items()
                          if v.default is inspect.Parameter.empty and v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD)]
                r["params_ok"] = all(p in uk for p in params)
                if "username" in uk and hasattr(st, "login_user"):
                    lp = [p for p in inspect.signature(st.login_user).parameters]
                    if all(p in uk for p in lp): st.login_user(**{p: uk[p] for p in lp})
                if r["params_ok"]: getattr(st, proc)(**{p: uk[p] for p in params})
                r["tool_runs"] = r["params_ok"]; r["tool_err"] = None if r["params_ok"] else "missing params"
            except Exception as e:
                r["tool_runs"] = False; r["tool_err"] = repr(e)[:120]
            try:
                lab, leaf = gt.leaf_results(dom, case, goal=proc)
                r["code_label"] = lab; r["fail_pat"] = list(fail_pattern(leaf)); r["leaf_err"] = None
            except Exception as e:
                r["code_label"] = None; r["leaf_err"] = repr(e)[:120]
            r["msg_missing"] = mentions(uk, msg)
            r["db_sig"] = json.dumps(db, sort_keys=True)
            rows.append(r)
    json.dump(rows, open(os.path.join(HERE, "validation_rows_" + (sys.argv[1] if len(sys.argv)>1 else "gen_full.json").replace(".json","") + ".json"), "w"))
    return rows

def real_stats():
    out = {}
    for dom, proc in gen.SEL:
        cs, same = gen.modal_variant(dom, proc)
        pats_same = collections.Counter(); pats_all = collections.Counter(); leaves_all = set(); leaves_same = set()
        sig = set()
        for c in cs:
            lab, leaf = gt.leaf_results(dom, c, goal=proc)
            names = lambda L: set(k.split("{")[0] for k in L)
            p = fail_pattern(leaf)
            pats_all[tuple(sorted(set(k.split("{")[0] for k in p)))] += 1
            sig.add(json.dumps(c["initial_database"], sort_keys=True))
            if c in same: pats_same[p] += 1
        out[(dom, proc)] = {"n_real": len(cs), "n_real_same": len(same), "perform": sum(c["action_should_succeed"] for c in cs),
                            "distinct_pat_same": len(pats_same), "distinct_pat_all_by_name": len(pats_all), "distinct_db": len(sig),
                            "pats_same": {" | ".join(k.split("{")[0] for k in p) or "-": n for p, n in pats_same.items()}}
    return out

if __name__ == "__main__":
    rows = run()
    json.dump({f"{k[0]}.{k[1]}": v for k, v in real_stats().items()}, open(os.path.join(HERE, "real_stats.json"), "w"), indent=1)
    print(len(rows))
