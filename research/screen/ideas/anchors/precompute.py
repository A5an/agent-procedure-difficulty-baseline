import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os, sys, time, random
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("LLM_BACKEND", "vertex")
import subprocess
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    os.environ["GOOGLE_CLOUD_PROJECT"] = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True).stdout.strip()
from alib import *
import run_baseline as RB, llm
os.chdir(ZR.SCREEN + "/ideas/anchors")

LIMIT = int(os.environ.get("LIMIT", 0))  # debug: first N prompts only


def enumerate_prompts(modes=("sim",)):
    allp = {}
    for bench in ["sopbench", "tau2"]:
        tasks = pd.read_csv(RB.DATA / bench / "tasks.csv")
        for scheme in ["new_procedures", "new_domain"]:
            seeds = [0] if scheme == "new_domain" else list(range(RB.N_SEEDS))
            for seed in seeds:
                folds = RB.splits(tasks, scheme, seed)
                for k, (tr, te) in enumerate(folds):
                    d = RB.DATA / bench
                    data = RB.load_dataset_for_fold(
                        abilities_path=d / "irt/1d_1pl/abilities.csv", items_path=d / "irt/1d_1pl/items.csv",
                        responses_path=d / "responses.jsonl", train_tasks=tr, test_tasks=te, fold_idx=k,
                        k_folds=len(folds), split_seed=seed, irt_cache_dir=RB.RESULTS / bench / "irt_splits" / scheme)
                    for mode in modes:
                        if mode == "random" and not (bench == "sopbench" and scheme == "new_domain"):
                            continue
                        _, test, *_r, prompts = fold_prompts(bench, tasks, data, mode, 0)
                        for t, p in prompts.items():
                            allp[phash(p)] = (p, bench, scheme, t, mode)
    return allp


def call(item):
    h, (p, *_rest) = item
    try:
        text, usage = llm.generate(p, max_tokens=1000, thinking="low")
    except Exception as e:  # noqa
        print("ERR", type(e).__name__, str(e)[:100], flush=True); return 0
    cache_put(h, text); return 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sim"
    allp = enumerate_prompts((mode,))
    if mode == "random":  # subsample 300 test tasks, seeded
        keys = sorted(allp, key=lambda h: allp[h][3]); rng = random.Random(0); sel = set(rng.sample(keys, 300))
        # sample by task id so the same tasks can be compared with anchors
        ids = sorted({allp[h][3] for h in allp}); rng = random.Random(0); sub = set(rng.sample(ids, 300))
        allp = {h: v for h, v in allp.items() if v[3] in sub}
    cache = load_cache()
    todo = [(h, v) for h, v in allp.items() if h not in cache]
    print(f"{mode}: {len(allp)} unique prompts, {len(todo)} to call", flush=True)
    if LIMIT: todo = todo[:LIMIT]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(call, todo):
            done += 1
            if done % 200 == 0: print(done, f"{time.time()-t0:.0f}s", flush=True)
    print("done", done, f"{time.time()-t0:.0f}s")
