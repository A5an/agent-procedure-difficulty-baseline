"""Condition-level targets from step outcomes (skip = 1 - step_correct), per unit and condition, global and per agent family.
Used only for training rows inside folds (the case list of each unit lets the fold code pick training cases)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, numpy as np, re, json
S = ZR.SCREEN
FLAG = {"gpt-5-mode_fc", "gpt-5-mini-mode_fc", "gemini-2.5-pro-mode_fc"}
def family(a):
    if a.startswith("claude-3-5"): return "claude35"
    if a.startswith("claude-3-7"): return "claude37"
    if a.startswith("gemini"): return "gemini"
    if a.startswith("gemma"): return "gemma"
    if a.startswith("gpt-4"): return "gpt4"
    if a.startswith(("o4", "deepseek", "gpt-5")): return "reasoning"
    if a.startswith("llama"): return "llama"
    if a.startswith("qwen2.5"): return "qwen25"
    if a.startswith("qwen3.5"): return "qwen35"
    raise ValueError(a)
FAMS = ["claude35", "claude37", "gemini", "gemma", "gpt4", "reasoning", "llama", "qwen25", "qwen35"]
def build():
    so = pd.read_csv(f"{S}/ideas2/steps/step_outcomes.csv.gz")
    so = so[~so.agent.isin(FLAG) & (so.step_type != "action")].copy()
    so["skip"] = 1 - so.step_correct
    so["fam"] = so.agent.map(family)
    cd = pd.read_csv("conds.csv")
    st = pd.read_csv(f"{S}/ideas2/steps/steps.csv", keep_default_na=False)
    un = json.load(open("units.json"))
    c2u = {c: u for u, cs in un.items() for c in cs}
    so["unit"] = so.case_id.map(c2u)
    # which condition (top-level step) does a step row belong to: top-level step_id itself, or the or_group of its leaf
    top = st[(st.step_type != "action")][["case_id", "step_id", "in_or", "group"]]
    top["cond_step"] = np.where(top.in_or == 1, top.group, top.step_id)
    so = so.merge(top[["case_id", "step_id", "cond_step"]], on=["case_id", "step_id"])
    # keep only rows of the top-level step itself (leaf or or_group) as the condition outcome
    so = so[so.step_id == so.cond_step]
    sh = pd.read_csv(ZR.REPO + "/data/sopbench/tasks.csv")[["case_id", "should_succeed"]]
    so = so.merge(sh, on="case_id")
    so = cd[["unit", "idx", "step_id"]].merge(so, on=["unit", "step_id"])
    so.to_csv("cond_outcomes.csv.gz", index=False)
    print(len(so), so.skip.mean())
    print(so.groupby("should_succeed").skip.mean(), so.groupby("blocking").skip.mean())
if __name__ == "__main__":
    build()
