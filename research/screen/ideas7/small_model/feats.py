"""Raw jsonl -> feature csvs sm_sopbench.csv, sm_tau2.csv (index case_id). usage: feats.py"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from transformers import AutoTokenizer
HERE = Path(ZR.SCREEN + "/ideas7/small_model"); OUT = HERE / "out"
MODEL = ZR.DATA + "/models/qwen3-8b-base-mlx-8bit"
tok = AutoTokenizer.from_pretrained(MODEL)
lse = lambda a: float(np.logaddexp.reduce(np.asarray(a, float)))
def softmax(v): v = np.asarray(v, float); e = np.exp(v - v.max()); return e / e.sum()
def ent(p): p = p[p > 0]; return float(-(p * np.log(p)).sum())

def bullets(pol, nll):
    """per-rule mean NLL: split the policy into rule lines (bullets), token counts by separate encoding scaled to the true length"""
    lines = [l for l in pol.split("\n") if l.strip()]
    rules = [l for l in lines if l.lstrip().startswith("•")] or lines
    cnt = [len(tok.encode(l + "\n", add_special_tokens=False)) for l in lines]
    sc = len(nll) / max(1, sum(cnt)); pos = 0; out = []
    for l, c in zip(lines, cnt):
        n = max(1, int(round(c * sc)))
        if l in rules: out.append(float(np.mean(nll[pos:pos + n])) if pos < len(nll) else np.nan)
        pos += n
    return [x for x in out if x == x]

rows = []
for l in open(OUT / "sop_raw.jsonl"):
    r = json.loads(l); lp = r["lp"]; goal = r["goal"]; chk = [c for c in r["req"] if c in lp]
    allc = [goal] + chk; v = np.array([lp[c] for c in allc]); p = softmax(v)
    d = {"case_id": r["case_id"], "sm_nchk": len(chk), "sm_tgt_lp": lp[goal]}
    if chk:
        cm = np.array([lp[c] for c in chk])
        d.update(sm_margin=lp[goal] - cm.max(), sm_margin_mean=lp[goal] - cm.mean(), sm_ent=ent(p), sm_ptgt=float(p[0]),
                 sm_rank=int((v > v[0]).sum() + 1), sm_rank_frac=float((v > v[0]).sum() / len(chk)))
    else: d.update(sm_margin=0.0, sm_margin_mean=0.0, sm_ent=0.0, sm_ptgt=1.0, sm_rank=1, sm_rank_frac=0.0)
    nl = np.array(r["nll_policy"]); b = bullets(r["policy"], nl)
    d.update(sm_pol_nll=float(nl.mean()), sm_pol_nll_max=float(max(b)) if b else float(nl.mean()), sm_pol_nll_bul_mean=float(np.mean(b)) if b else float(nl.mean()),
             sm_pol_nll_sum=float(nl.sum()), sm_msg_nll=float(np.mean(r["nll_msg"])))
    rows.append(d)
sop = pd.DataFrame(rows).set_index("case_id"); sop.to_csv(HERE / "sm_sopbench.csv"); print(sop.describe().T[["count", "mean", "std"]])

rows = []
for l in open(OUT / "tau2_raw.jsonl"):
    r = json.loads(l); lp = r["lp"]; ty = r["types"]; names = list(lp); v = np.array([lp[n] for n in names]); p = softmax(v)
    w = [lp[n] for n in names if ty[n] == "write"]; rd = [lp[n] for n in names if ty[n] == "read"]
    top = names[int(v.argmax())]
    rows.append({"case_id": r["case_id"], "sm_wr_margin": (lse(w) - lse(rd)) if w and rd else 0.0, "sm_ent": ent(p), "sm_pmax": float(p.max()),
                 "sm_top_write": float(ty[top] == "write"), "sm_p_write": float(sum(p[i] for i, n in enumerate(names) if ty[n] == "write")),
                 "sm_p_xfer": float(sum(p[i] for i, n in enumerate(names) if "transfer" in n)), "sm_msg_nll": float(np.mean(r["nll_msg"])),
                 "sm_best_write_vs_read": (max(w) - max(rd)) if w and rd else 0.0})
t2 = pd.DataFrame(rows).set_index("case_id"); t2.to_csv(HERE / "sm_tau2.csv"); print(t2.describe().T[["count", "mean", "std"]])
