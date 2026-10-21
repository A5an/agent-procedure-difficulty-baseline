import os, json, random
from alib import *
import run_baseline as RB
from scipy.stats import spearmanr
os.chdir(HERE); cache = load_cache(); bench = "sopbench"
tasks = pd.read_csv(RB.DATA / bench / "tasks.csv"); items = pd.read_csv(RB.DATA / bench / "irt/1d_1pl/items.csv", index_col=0)
ids = sorted(tasks.case_id); sub = set(random.Random(0).sample(ids, 300))  # same rule as precompute (sorted ids of all test tasks)
rows = []
folds = RB.splits(tasks, "new_domain", 0)
for k, (tr, te) in enumerate(folds):
    d = RB.DATA / bench
    data = RB.load_dataset_for_fold(abilities_path=d/"irt/1d_1pl/abilities.csv", items_path=d/"irt/1d_1pl/items.csv", responses_path=d/"responses.jsonl",
        train_tasks=tr, test_tasks=te, fold_idx=k, k_folds=len(folds), split_seed=0, irt_cache_dir=RB.RESULTS/bench/"irt_splits"/"new_domain")
    P = {}
    for mode in ["sim", "random"]:
        train, test, share, beta, anc, prompts = fold_prompts(bench, tasks, data, mode, 0, only=sub)
        P[mode] = {t: parse(cache.get(phash(prompts[t]))) for t in test}
    for t in P["sim"]:
        rows.append(dict(case_id=t, fold=k, sim=P["sim"][t], rnd=P["random"][t], b=items.loc[t, "b"], domain=tasks.set_index("case_id").domain[t]))
df = pd.DataFrame(rows).dropna(); out = {"n": len(df)}
for c in ["sim", "rnd"]:
    v = [spearmanr(-x[c], x.b).correlation for _, x in df.groupby("domain") if len(x) > 5]  # difficulty direction
    out["within_domain_rho_" + c] = float(np.mean(v)); out["pooled_rho_" + c] = float(spearmanr(-df[c], df.b).correlation)
out["spearman_sim_vs_rnd"] = float(spearmanr(df.sim, df.rnd).correlation)
print(out); json.dump(out, open("rand_cmp.json", "w"))
