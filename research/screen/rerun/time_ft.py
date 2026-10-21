import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys,time,json,os
sys.path.insert(0,ZR.SCREEN); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import harness as H
from finetuned import FineTunedEncoder
import pandas as pd
b="tau2"; tasks=pd.read_csv(H.RB.DATA/b/"tasks.csv")
texts={json.loads(l)["case_id"]:json.loads(l)["text"] for l in (H.RB.DATA/b/"statements.jsonl").read_text().splitlines()}
tr,te=H.RB.splits(tasks,"new_procedures",0)[0]
d=H.RB.DATA/b
from experiment_new_tasks.dataset import load_dataset_for_fold
data=load_dataset_for_fold(abilities_path=d/"irt/1d_1pl/abilities.csv",items_path=d/"irt/1d_1pl/items.csv",responses_path=d/"responses.jsonl",train_tasks=tr,test_tasks=te,fold_idx=0,k_folds=5,split_seed=0,irt_cache_dir=H.RB.RESULTS/b/"irt_splits"/"new_procedures")
m=FineTunedEncoder(texts,seed=0); t=time.time(); m.fit(data,tr); print("fit",len(tr),time.time()-t)
a=data.train_abilities.index[0]; print(m.predict_probability(data,a,te[0]))
