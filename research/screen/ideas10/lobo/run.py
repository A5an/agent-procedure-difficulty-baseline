"""Pre-registered LOBO run (PREREG.md). usage: run.py"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json
sys.path.insert(0, ZR.SCREEN + "/ideas10/common")
sys.path.insert(0, ZR.SCREEN + "/ideas9/gap")
import pandas as pd, numpy as np
import lobo2
from feats import GAP
S = ZR.SCREEN
B = ["sopbench", "tau2", "tauk_banking"]
def tab(p): d = pd.read_csv(p); return d.set_index(d.columns[0]).apply(pd.to_numeric, errors="coerce")
rep = {k: lobo2.repo_feat(k) for k in ["length", "adele", "human_time"]}
gap = {b: tab(f"{S}/ideas9/gap/gap_{b}.csv")[GAP] for b in B}
pri = {b: tab(f"{S}/ideas10/prior/prior_{b}.csv") for b in B}
plan = {"sopbench": tab(f"{S}/ideas3/plan_tools/features/plan3_sopbench.csv"), "tau2": tab(f"{S}/ideas3/plan_tools/features/plan3_tau2.csv"),
        "tauk_banking": tab(f"{S}/ideas7/evo_check/bank_plan.csv")}
rub = {b: tab(f"{S}/rubric/features/rub_proc_{b}.csv").select_dtypes("number") for b in B}
fc = {b: tab(f"{S}/ideas10/fc/fc_{b}.csv") for b in B}
def cat(*ds): return {b: pd.concat([d[b] for d in ds], axis=1) for b in B}
M = {"length": rep["length"], "adele": rep["adele"], "human_time": rep["human_time"],
     "L_gap": cat(rep["adele"], gap), "L_prior": cat(rep["adele"], pri),
     "L_doc": cat(rep["adele"], plan, gap, pri, rub)}
M["L_doc_len"] = cat(M["L_doc"], rep["length"], rep["human_time"])
M["L_doc_fc"] = cat(M["L_doc_len"], fc)
for k, v in M.items():
    for b in B: print(k, b, v[b].shape, int(v[b].reindex(lobo2.tasks().query("bench == @b").case_id).isna().any(axis=1).sum()), "rows with NaN")
res = lobo2.run(M)
json.dump(res, open("lobo_result.json", "w"), indent=1); lobo2.show(res)
