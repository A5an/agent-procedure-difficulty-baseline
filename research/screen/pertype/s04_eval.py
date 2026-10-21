"""Harness evaluation + within-agent AUC / log-loss with paired procedure bootstrap."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
import harness as H
from scipy.stats import rankdata
OUT = ZR.SCREEN + "/pertype/run"
NEW = {"share_none": "(ii) null: agent's overall share, no types", "share_fam": "(ii) shrunk shares, action family",
 "share_tree": "(ii) shares, rule-tree type", "share_dom": "(ii) shares, domain", "share_famtree": "(ii) shares, family x tree",
 "logit_fam": "(iii) agent x family logistic on baseline", "logit_tree": "(iii) agent x tree", "logit_dom": "(iii) agent x domain",
 "logit_all": "(iii) agent x (family+tree+domain)", "mirt_q3": "(iv) confirmatory MIRT, 3 fixed dims",
 "logit_all_l10": "(iii) sensitivity: lambda fixed 10", "logit_all_l30": "(iii) sensitivity: lambda fixed 30", "mirt_q3_l10": "(iv) sensitivity: lambda fixed 10",
 "main_all": "control: agent intercept + shared type effects (no interaction)", "inter_all": "(iii') stand-alone: intercept + shared type effects + agent x type",
 "typebeta": "control: one-hot types -> ridge difficulty", "adele_onehot": "control: ADeLe + one-hot types -> ridge difficulty"}
if __name__ == "__main__":
    out = H.evaluate_all(OUT, NEW)
    json.dump(out, open(OUT + "/eval_copy.json", "w"))
