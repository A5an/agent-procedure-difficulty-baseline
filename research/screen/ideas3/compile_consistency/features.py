"""Per-unit consistency features from runs.json and the program texts. No labels, no real records."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, ast, itertools, collections, math
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_programs import FILES
runs = json.load(open(HERE + "/runs.json"))
progs = [json.load(open(f)) for f in FILES]
def jac_d(a, b):
    a, b = set(a), set(b); u = a | b
    return 0.0 if not u else 1 - len(a & b) / len(u)
def code_stats(code):
    try: tree = ast.parse(code)
    except SyntaxError: return None
    fields, conds, tools_ref, br = set(), set(), set(), 0
    for n in ast.walk(tree):
        if isinstance(n, (ast.If, ast.IfExp)): br += 1
        if isinstance(n, ast.BoolOp): br += len(n.values) - 1
        if isinstance(n, ast.Subscript):
            s = n.slice
            if isinstance(s, ast.Constant) and isinstance(s.value, str): fields.add(s.value)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr == "get" and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                fields.add(n.args[0].value)
            if isinstance(n.func.value, ast.Name) and n.func.value.id == "tools": tools_ref.add(n.func.attr)
        if isinstance(n, ast.Compare):
            consts = []
            for x in [n.left] + n.comparators:
                if isinstance(x, ast.Constant) and not isinstance(x.value, (type(None),)): consts.append(repr(x.value))
            conds.add((tuple(type(o).__name__ for o in n.ops), tuple(sorted(consts))))
    lines = len([l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")])
    return {"fields": fields, "conds": conds, "tools_ref": tools_ref, "branches": br, "lines": lines}
def mean_pair(sets):
    ps = [jac_d(a, b) for a, b in itertools.combinations(sets, 2)]
    return float(np.mean(ps)) if ps else np.nan
def feats(k):
    r = runs[k]; ns = r["scen"]
    P = [(i, p) for i, p in enumerate(r["progs"]) if p is not None]
    cs = [code_stats((progs[i].get(k) or {}).get("code") or "") for i, _ in P]
    keep = [(i, p, c) for (i, p), c in zip(P, cs) if c is not None]
    f = {"n_prog": len(keep), "n_scen": ns}
    if len(keep) < 3: return f
    dec = np.array([[1 if p[s]["goal"] else 0 for s in range(ns)] for _, p, _ in keep])   # prog x scen
    share = dec.mean(0)
    f["dec_dis"] = float(np.mean((share > 0) & (share < 1)))
    f["dec_ent"] = float(np.mean([0 if x in (0, 1) else -(x * math.log2(x) + (1 - x) * math.log2(1 - x)) for x in share]))
    f["dec_pair"] = float(np.mean([np.mean(dec[a] != dec[b]) for a, b in itertools.combinations(range(len(keep)), 2)]))
    f["perform_rate"] = float(dec.mean())
    f["tool_pair"] = float(np.nanmean([mean_pair([p[s]["tools"] for _, p, _ in keep]) for s in range(ns)]))
    f["call_pair"] = float(np.nanmean([mean_pair([p[s]["calls"] for _, p, _ in keep]) for s in range(ns)]))
    f["crash"] = float(np.mean([p[s]["err"] is not None for _, p, _ in keep for s in range(ns)]))
    ncalls = sum(p[s]["n_calls"] for _, p, _ in keep for s in range(ns))
    f["tool_err"] = float(sum(p[s]["n_tool_err"] for _, p, _ in keep for s in range(ns)) / max(1, ncalls))
    allf = [c["fields"] for _, _, c in keep]
    f["n_fields"] = len(set().union(*allf)); f["fields_pair"] = mean_pair(allf)
    f["fields_only_one"] = len([x for x in set().union(*allf) if sum(x in a for a in allf) == 1])
    f["len_mean"] = float(np.mean([c["lines"] for _, _, c in keep])); f["len_cv"] = float(np.std([c["lines"] for _, _, c in keep]) / max(1, f["len_mean"]))
    f["branch_mean"] = float(np.mean([c["branches"] for _, _, c in keep])); f["branch_sd"] = float(np.std([c["branches"] for _, _, c in keep]))
    allc = [c["conds"] for _, _, c in keep]
    f["cond_pair"] = mean_pair(allc); f["cond_n_diff"] = len(set().union(*allc)) - len(set.intersection(*allc))
    allt = [c["tools_ref"] for _, _, c in keep]
    f["toolref_pair"] = mean_pair(allt); f["toolref_n_diff"] = len(set().union(*allt)) - len(set.intersection(*allt))
    return f
FIXED = ["dec_dis", "dec_ent", "tool_pair", "crash", "tool_err", "n_fields", "len_mean", "branch_mean", "cond_pair", "cond_n_diff"]
if __name__ == "__main__":
    import sop
    rows = {k: feats(k) for k in runs}
    df = pd.DataFrame(rows).T; df.index.name = "unit"; df.to_csv(HERE + "/unit_features.csv")
    ids = json.load(open(ZR.SCREEN + "/ideas2/compile/results_A.json"))
    cs = pd.DataFrame([{"case_id": r["case_id"], "unit": r["unit"]} for r in ids]).set_index("case_id")
    cf = cs.join(df, on="unit")
    # units with missing features get column medians (documented)
    n_missing = int(cf[FIXED].isna().any(axis=1).sum())
    for c in FIXED: cf[c] = cf[c].fillna(cf[c].median())
    cf[["unit"] + FIXED].reset_index().drop(columns="unit").to_csv(HERE + "/cons_sopbench.csv", index=False)
    print(df[FIXED].describe().T, "cases w/ imputed", n_missing)
