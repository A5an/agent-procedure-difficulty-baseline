"""python run_group.py <group> <bench>   -> out/<group>/<bench>/obs_<scheme>.csv.gz"""
import sys
from pathlib import Path
import pandas as pd
import harness as H
import mlib as L

HERE = Path(__file__).parent
group, bench = sys.argv[1], sys.argv[2]
FE = HERE / "features"
extra = {n: {b: FE / b / f"{n}.csv" for b in H.BENCHES} for n in ("adele_ht", "adele_ht_len")}
tasks = {b: pd.read_csv(H.RB.DATA / b / "tasks.csv") for b in H.BENCHES}
G = H.RB.GroupedRidgePredictor.SOURCE_ALPHA_GRIDS


def make(src, Mm, b):
    t = tasks[b]
    a = src["adele"]
    if group == "trees":
        return {"rf_adele": H.RB.DifficultyPredictorAdapter(L.TreeDifficulty(a, t, "rf")),
                "gb_adele": H.RB.DifficultyPredictorAdapter(L.TreeDifficulty(a, t, "gb"))}
    if group == "rank":
        return {"rank_adele": H.RB.DifficultyPredictorAdapter(L.RankDifficulty(a, t))}
    if group == "irt":
        return {"mirt_adele_d2": Mm.AmortizedMIRT(a, name="mirt_adele_d2", dims=2),
                "mirt_adele_d4": Mm.AmortizedMIRT(a, name="mirt_adele_d4", dims=4),
                "lltm_adele_ht": Mm.AmortizedIRT(src["adele_ht"], name="lltm_adele_ht"),
                "lltm_adele_ht_len": Mm.AmortizedIRT(src["adele_ht_len"], name="lltm_adele_ht_len")}
    if group == "ridge":
        return {"ridge_adele_ht": H.ridge(src["adele_ht"]),
                "ridge_adele_ht_len": H.ridge(src["adele_ht_len"]),
                "grp_adele_ht": H.grouped([a, src["human_time"]], {"adele": G["LLM Judge"], "human_time": G["Embedding"]}),
                "grp_adele_ht_lowgrid": H.grouped([a, src["human_time"]], {"adele": G["LLM Judge"], "human_time": G["LLM Judge"]})}
    if group == "agent":
        return {"agent_lr_adele": L.PerAgentLogit(a, t, "agent_lr_adele")}
    if group == "shrunk":
        return {"agent_shrunk_adele": L.ShrunkPerAgent(a, t, "agent_shrunk_adele")}
    raise ValueError(group)


H.run_all(HERE / "out" / group, make, extra_csv=extra, benches=[bench])
