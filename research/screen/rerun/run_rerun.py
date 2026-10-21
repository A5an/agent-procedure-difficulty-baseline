"""Rerun of previously screened methods through the shared harness (baseline protocol, seeded folds).
PART=singles|ft|combos selects what is run; output dirs are separate and merged by evaluate_rerun.py."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, os, sys
from pathlib import Path
sys.path.insert(0, ZR.SCREEN)
import harness as H

HERE = Path(__file__).parent
FEAT = HERE / "features"
REPO_DATA = H.REPO / "data"
PART = os.environ.get("PART", "singles")
SCH = ["new_procedures", "new_domain"]
BENCH_ALL = ["sopbench", "tau2", "tauk_banking"]

def csv(name):
    return {b: FEAT / b / f"{name}.csv" for b in BENCH_ALL if (FEAT / b / f"{name}.csv").exists()}
def npz(name):
    return {b: H.EmbeddingFeatureSource(FEAT / b / f"{name}.npz", name=name) for b in BENCH_ALL}

csv_src = {n: csv(n) for n in ["entropy", "bpm_complexity", "llm_forecast"]}
emb_src = {"probe_L18": npz("probe_L18"), "probe_L36": npz("probe_L36")}

if PART == "singles":
    def make(src, M, bench):
        p = {n: H.ridge(src[n]) for n in ["probe_L18", "probe_L36", "entropy", "llm_forecast", "emb_bge"]}
        if "bpm_complexity" in src:
            p["bpm_complexity"] = H.ridge(src["bpm_complexity"])
        p["amortized_irt_bge"] = M.AmortizedIRT(src["emb_bge"], name="amortized_irt_bge")
        return p
    H.run_all(HERE / "out_singles", make, extra_csv=csv_src, extra_sources=emb_src, schemes=SCH)
elif PART == "ft":
    sys.path.insert(0, str(HERE))
    from finetuned import FineTunedEncoder
    texts = {}
    for b in BENCH_ALL:
        texts[b] = {}
        for l in (REPO_DATA / b / "statements.jsonl").read_text().splitlines():
            r = json.loads(l); texts[b][r["case_id"]] = r["text"]
    def make(src, M, bench):
        return {"finetuned_bge_small": FineTunedEncoder(texts[bench], seed=0)}
    H.run_all(HERE / "out_ft", make, schemes=SCH)
elif PART == "combos":
    from itertools import product
    names = os.environ.get("CAT", "probe_L36,emb_bge").split(",")
    def make(src, M, bench):
        g = H.RB.GroupedRidgePredictor.SOURCE_ALPHA_GRIDS
        return {f"cat_adele_{n}": H.grouped([src[n], src["adele"]], {n: g["Embedding"], "adele": g["LLM Judge"]}) for n in names}
    H.run_all(HERE / "out_combos", make, extra_sources=emb_src, schemes=SCH)
