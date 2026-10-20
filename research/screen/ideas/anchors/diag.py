import os, sys, json
from alib import *
import run_baseline as RB
from scipy.stats import spearmanr
os.chdir(HERE)
cache = load_cache()
rows = []
for bench in ["sopbench", "tau2"]:
    tasks = pd.read_csv(RB.DATA / bench / "tasks.csv")
    items = pd.read_csv(RB.DATA / bench / "irt/1d_1pl/items.csv", index_col=0)
    full = {}
    for l in open(RB.DATA / bench / "responses.jsonl"):
        r = json.loads(l)
        for t, v in r["responses"].items():
            k, n = (v["successes"], v["trials"]) if isinstance(v, dict) else (int(v), 1)
            a = full.setdefault(t, [0, 0]); a[0] += k; a[1] += n
    for scheme in ["new_procedures", "new_domain"]:
        seeds = [0] if scheme == "new_domain" else list(range(RB.N_SEEDS))
        for seed in seeds:
            folds = RB.splits(tasks, scheme, seed)
            for k, (tr, te) in enumerate(folds):
                d = RB.DATA / bench
                data = RB.load_dataset_for_fold(abilities_path=d / "irt/1d_1pl/abilities.csv", items_path=d / "irt/1d_1pl/items.csv",
                    responses_path=d / "responses.jsonl", train_tasks=tr, test_tasks=te, fold_idx=k, k_folds=len(folds),
                    split_seed=seed, irt_cache_dir=RB.RESULTS / bench / "irt_splits" / scheme)
                train, test, share, beta, anc, prompts = fold_prompts(bench, tasks, data)
                for t in test:
                    v = parse(cache.get(phash(prompts[t])))
                    rows.append(dict(bench=bench, scheme=scheme, seed=seed, fold=k, case_id=t, llm=v,
                                     near_share=np.mean([share[x] for x in anc[t][0]]) * 100,
                                     true_share=100 * full[t][0] / full[t][1], b=items.loc[t, "b"]))
df = pd.DataFrame(rows).merge(pd.concat([pd.read_csv(RB.DATA / b / "tasks.csv").assign(bench=b) for b in ["sopbench", "tau2"]])[["bench", "case_id", "domain", "procedure_id"]], on=["bench", "case_id"])
df.to_csv(HERE / "diag_rows.csv", index=False)
out = {}
for (b, s), g in df.groupby(["bench", "scheme"]):
    g = g.dropna(subset=["llm"])
    o = {"n": len(g), "unparsed": int(df[(df.bench == b) & (df.scheme == s)].llm.isna().sum())}
    o["spearman_llm_vs_near_share"] = float(np.mean([spearmanr(x.llm, x.near_share).correlation for _, x in g.groupby("seed")]))
    o["spearman_llm_vs_true_share_pooled"] = float(np.mean([spearmanr(x.llm, x.true_share).correlation for _, x in g.groupby("seed")]))
    o["spearman_near_vs_true_share_pooled"] = float(np.mean([spearmanr(x.near_share, x.true_share).correlation for _, x in g.groupby("seed")]))
    # within-domain (protocol-like) rho, success-rate direction
    def wd(col):
        v = []
        for _, x in g.groupby(["seed", "fold", "domain"]):
            if len(x) > 3 and x[col].nunique() > 1: v.append(spearmanr(x[col], x.true_share).correlation)
        return float(np.mean(v))
    o["within_domain_rho_llm"] = wd("llm"); o["within_domain_rho_near"] = wd("near_share")
    if b == "sopbench":  # within procedure
        v = []
        for _, x in g[g.seed == 0].groupby("procedure_id"):
            if len(x) > 3 and x.llm.nunique() > 1: v.append(spearmanr(x.llm, x.true_share).correlation)
        o["within_procedure_rho_llm_mean"] = float(np.mean(v)); o["n_procedures"] = len(v)
        v = []
        for _, x in g[g.seed == 0].groupby("procedure_id"):
            if len(x) > 3 and x.near_share.nunique() > 1: v.append(spearmanr(x.near_share, x.true_share).correlation)
        o["within_procedure_rho_near_mean"] = float(np.mean(v))
    g0 = g[g.seed == 0].copy()
    g0["dec"] = pd.qcut(g0.llm.rank(method="first"), 10, labels=False)
    o["calibration_deciles"] = [[round(x.llm.mean(), 1), round(x.true_share.mean(), 1)] for _, x in g0.groupby("dec")]
    out[f"{b}|{s}"] = o
(HERE / "diag.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
