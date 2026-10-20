import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys; sys.path.insert(0,ZR.SCREEN); sys.path.insert(0,ZR.SCREEN + "/ideas")
import harness as H, lupi_lib as L, time, pandas as pd
from cpulock import cpu_lock
bench="tau2"
tasks=pd.read_csv(H.RB.DATA/bench/"tasks.csv"); src=H.RB.sources(bench)
with cpu_lock("lupi"):
    for scheme in ["new_procedures"]:
        folds=H.RB.splits(tasks,scheme,0)[:1]
        import experiment_new_tasks.dataset as D
        tr,te=folds[0]; res=H.RB.RESULTS/bench
        data=D.load_dataset_for_fold(abilities_path=H.RB.DATA/bench/"irt/1d_1pl/abilities.csv",items_path=H.RB.DATA/bench/"irt/1d_1pl/items.csv",responses_path=H.RB.DATA/bench/"responses.jsonl",train_tasks=tr,test_tasks=te,fold_idx=0,k_folds=5,split_seed=0,irt_cache_dir=res/"irt_splits"/scheme)
        for m in ["x_ridge","lupi","mix2","lupi_mix2"]:
            t0=time.time(); p=L.Base(bench,m); p.fit(data,tr)
            a=data.train_abilities.index[0]; print(m,p.mix if m!="x_ridge" else "",p.predict_probability(data,a,te[0]),round(time.time()-t0,1))
