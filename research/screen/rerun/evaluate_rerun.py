"""Merge singles, concatenated-ridge combos and fine-tuned encoder columns (identical folds/rows) and score with the repo evaluator."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, shutil, json
from pathlib import Path
sys.path.insert(0, ZR.SCREEN)
import harness as H
import pandas as pd
HERE = Path(__file__).parent
KEYS = ["seed", "fold", "agent", "case_id", "y"]
FT = (HERE / "out_ft" / "sopbench" / "obs_new_procedures.csv.gz").exists() and os.environ.get("USE_FT", "1") == "1"
LOGIT_COMBOS = [c.split("+") for c in os.environ.get("LOGIT", "probe_L36,emb_bge").split(",") if c]
out = HERE / "out_final"
for b in ["sopbench", "tau2"]:
    (out / b).mkdir(parents=True, exist_ok=True)
    for s in H.SCHEMES[1:]:
        m = pd.read_csv(HERE / "out_singles" / b / f"obs_{s}.csv.gz")
        for extra in (["out_combos"] + (["out_ft"] if FT else [])):
            p = HERE / extra / b / f"obs_{s}.csv.gz"
            if not p.exists(): continue
            e = pd.read_csv(p)
            assert len(e) == len(m) and (e[KEYS].values == m[KEYS].values).all(), (extra, b, s)
            assert (e["adele"] - m["adele"]).abs().max() < 1e-9
            m = pd.concat([m, e[[c for c in e.columns if c not in m.columns]]], axis=1)
        m.to_csv(out / b / f"obs_{s}.csv.gz", index=False)
new = {"probe_L18": "hidden state L18 ridge", "probe_L36": "hidden state L36 ridge", "entropy": "token entropy ridge",
       "llm_forecast": "PRECOG-style forecast ridge", "emb_bge": "bge-base ridge", "amortized_irt_bge": "amortized IRT on bge",
       "cat_adele_probe_L36": "concat ridge adele+L36", "cat_adele_emb_bge": "concat ridge adele+bge"}
if FT: new["finetuned_bge_small"] = "fine-tuned bge-small"
combos = [(a, "adele") for a in [c[0] for c in LOGIT_COMBOS] + (["finetuned_bge_small"] if FT and os.environ.get("FT_COMBO") == "1" else [])]
r = H.evaluate_all(out, new, combos=combos)
# SOPBench-only evaluation for the BPM metrics (tau2/tauk have no per-task process)
sop = HERE / "out_sop"; (sop / "sopbench").mkdir(parents=True, exist_ok=True)
for s in H.SCHEMES[1:]:
    m = pd.read_csv(HERE / "out_singles" / "sopbench" / f"obs_{s}.csv.gz")
    m.to_csv(sop / "sopbench" / f"obs_{s}.csv.gz", index=False)
H.E.NEW = ["sopbench|new_procedures", "sopbench|new_domain"]
H.evaluate_all(sop, {"bpm_complexity": "BPM complexity", "probe_L36": "x", "emb_bge": "y"}, combos=[("bpm_complexity", "adele")])
