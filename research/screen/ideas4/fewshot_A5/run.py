"""usage: run.py tau2|sop. SEEDS env sets seeds. See PREREG.md."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, importlib.util
S = ZR.SCREEN
which = sys.argv[1]
sys.argv = [sys.argv[0], "main"]
sys.path.insert(0, S); sys.path.insert(0, S + "/ideas"); sys.path.insert(0, S + "/rubric"); sys.path.insert(0, S + "/ideas3/conditions")
import numpy as np, pandas as pd
from pathlib import Path
spec = importlib.util.spec_from_file_location("combo_run", S + "/ideas3/combo/run.py"); C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
import harness as H
from cpulock import cpu_lock
sys.path.insert(0, str(Path(__file__).parent)); import a5
HERE = Path(__file__).parent
_FC = {}
def fold_tabs(bench, tr, te):
    key = (bench, hash(tuple(sorted(tr))), hash(tuple(sorted(te))))
    if key not in _FC:
        A = a5.stage_a(bench, tr, te); D = a5.stage_a(bench, tr, te, demean=True)
        _FC[key] = {"A": A, "Adm": D.rename(columns=lambda c: "dm_" + c), "Aw": A[["writes"]].rename(columns=lambda c: "w_" + c),
                    "Aall": A[["all"]].rename(columns=lambda c: "all_" + c), "Alen": A[["len"]].rename(columns=lambda c: "len_" + c)}
    return _FC[key]
class GF(C.Grouped):
    def fit(self, data, tr):
        self.tabs = dict(self.tabs); self.tabs.update(fold_tabs(self.bench, list(tr), list(data.test_tasks)))
        super().fit(data, tr)
class LadF(C.Ladder):
    def __init__(self, bench, fam, base, rub, name):
        super().__init__(2, C.TYPES, C.GLOB, rub, extra_table=None, name=name); self.bench, self.fam, self.base = bench, fam, base
    def fit(self, data, tr):
        T = fold_tabs(self.bench, list(tr), list(data.test_tasks))
        ids = list(tr) + list(data.test_tasks)
        self.extra = pd.concat([self.base] + [T[f] for f in self.fam], axis=1).loc[ids]; self._test = None
        super().fit(data, tr)
def make_preds(src, M, bench):
    t = C.static_tables(bench); sop = bench == "sopbench"
    ex = pd.concat([t["ADELE"], t["PLAN"]] + ([t["CMP"]] if sop else []), axis=1)
    p = {"A": GF("A", bench, ["A"]), "A_adele": GF("A_adele", bench, ["ADELE", "A"]),
         "c_lad": C.Ladder(2, C.TYPES, C.GLOB, t["RUB"], extra_table=ex, name="c_lad"),
         "c_lad_A": LadF(bench, ["A"], ex, t["RUB"], "c_lad_A")}
    if not sop:
        p.update({"A_dm": GF("A_dm", bench, ["Adm"]), "A_w": GF("A_w", bench, ["Aw"]), "A_all": GF("A_all", bench, ["Aall"]),
                  "A_len": GF("A_len", bench, ["Alen"]), "A_raw": GF("A_raw", bench, ["PLAN"])})
    return p
if which == "tau2":
    out = HERE / "out_tau2"; benches = ["tau2"]
else:
    out = HERE / "out_sop"; benches = ["sopbench"]
with cpu_lock("fewshot_A5"):
    H.run_all(out, make_preds, benches=benches, schemes=["new_procedures", "new_domain"])
# audit: every gold lookup was a training task of its fold is asserted inside gold_of; report the count
print("gold lookups", len(a5.LOOKUPS), "run done")
