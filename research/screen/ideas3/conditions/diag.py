import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, re
sys.path.insert(0, ZR.SCREEN); sys.path.insert(0, ZR.SCREEN + "/ideas")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from cpulock import cpu_lock
import cm
from cm import *
import harness as H
import run_baseline as RB
tk = pd.read_csv(f"{REPO}/data/sopbench/tasks.csv")
allc = list(tk.case_id)
GROUPS = {
 "position": ["pos", "pos_rel", "is_first", "is_last", "from_end", "n_cond"],
 "length": ["nchar", "nword", "nsent", "lognchar", "n_commas", "n_parens", "n_and", "n_or_word"],
 "names_field_tool": ["n_quoted_fields", "n_distinct_fields", "has_field", "tool_name_overlap", "says_database", "says_account", "exists_style", "must_have_style"],
 "negation": ["neg", "n_neg", "bold_neg", "n_bold"],
 "or_group": ["in_or", "n_alt", "nested", "seq"],
 "numeric": ["n_numbers", "has_number", "n_compare", "has_compare"],
 "date": ["date_words", "has_date", "date_arith_rx"],
 "cross_ref": ["ref_words", "has_ref", "n_other_cond_shared_fields", "is_auth_text"],
 "llm_tags": [c for c in cm.FEATS if c.startswith("llm_")],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(cm.FEATS), set(cm.FEATS) ^ set(sum(GROUPS.values(), []))
out = {}
with cpu_lock("conditions_diag2"):
    # 1. full-data coefficients
    m = CondModel(allc)
    for name in ["perf", "fail"]:
        r, mu, sd, agg, alpha = m.models[name]
        co = pd.Series(r.coef_, index=cm.FEATS).sort_values()
        out[f"coef_{name}"] = dict(alpha=alpha, top_neg=co.head(8).round(4).to_dict(), top_pos=co.tail(8).round(4).to_dict())
        y = agg["mean"].values
        X = cm.cd.loc[agg.row, cm.FEATS]
        uni = {f: spearmanr(X[f], y)[0] for f in cm.FEATS if X[f].nunique() > 1}
        u = pd.Series(uni).sort_values()
        out[f"uni_{name}"] = dict(neg=u.head(8).round(3).to_dict(), pos=u.tail(8).round(3).to_dict())
        # within step type (type is a diagnostic from the tree)
        st = cm.cd.loc[agg.row, "step_type"].values
        yr = y - pd.Series(y).groupby(st).transform("mean").values
        wi = {}
        for f in cm.FEATS:
            x = X[f].values.astype(float)
            xr = x - pd.Series(x).groupby(st).transform("mean").values
            if np.std(xr) > 1e-9: wi[f] = spearmanr(xr, yr)[0]
        w = pd.Series(wi).sort_values()
        out[f"within_type_{name}"] = dict(neg=w.head(6).round(3).to_dict(), pos=w.tail(6).round(3).to_dict())
    # 2. novelty and group ablation at condition level
    def norm(t): return re.sub(r"\d+", "#", re.sub(r'"[^"]*"', "Q", t.lower()))
    cd["norm"] = cd.text.map(norm)
    ORIG = list(cm.FEATS)
    def run(featset, label):
        cm.FEATS[:] = featset
        res = {"perf": [], "fail": []}
        for scheme in ["new_procedures", "new_domain"]:
            seeds = [0] if scheme == "new_domain" else range(RB.N_SEEDS)
            for seed in seeds:
                for k, (tr, te) in enumerate(RB.splits(tk, scheme, seed)):
                    mm = CondModel(tr)
                    otr = OUT[OUT.case_id.isin(set(tr))]
                    seen = set(cd.loc[[ROW[(u, i)] for u, i in set(zip(otr.unit, otr.idx))], "norm"])
                    o = OUT[OUT.case_id.isin(set(te))]
                    for name, sub in [("perf", o[o.should_succeed == 1]), ("fail", o[o.blocking == 1])]:
                        a = sub.groupby(["unit", "idx"]).skip.agg(["mean", "size"]).reset_index()
                        a["row"] = [ROW[(u, i)] for u, i in zip(a.unit, a.idx)]
                        a["p"] = mm.pred[name][a.row]; a["novel"] = [cd.norm[r] not in seen for r in a.row]
                        a["scheme"] = scheme; a["fold"] = f"{seed}_{k}"
                        res[name].append(a)
        o2 = {}
        for name in res:
            D = pd.concat(res[name])
            r_all = np.mean([spearmanr(g.p, g["mean"])[0] for _, g in D.groupby(["scheme", "fold"]) if g.p.nunique() > 1])
            o2[name] = dict(rho=round(float(r_all), 3), share_novel=round(float(D.novel.mean()), 3))
            Dn = D[D.novel]
            if len(Dn) > 30:
                o2[name]["rho_novel_text_pooled"] = round(float(spearmanr(Dn.p, Dn["mean"])[0]), 3)
                o2[name]["rho_seen_text_pooled"] = round(float(spearmanr(D[~D.novel].p, D[~D.novel]["mean"])[0]), 3)
        out[label] = o2
        print(label, o2, flush=True)
    run(ORIG, "full")
    for g, fs in GROUPS.items():
        run([f for f in ORIG if f not in fs], f"drop_{g}")
        run(fs, f"only_{g}")
    cm.FEATS[:] = ORIG
json.dump(out, open("diag.json", "w"), indent=1)
