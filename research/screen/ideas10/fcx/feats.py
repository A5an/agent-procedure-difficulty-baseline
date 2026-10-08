"""Build FCX: z-scored average of fc_diff (and the 5 FC features) over sources S1 (fc/), S2 (son_raw), G1, G2 (gem_raw).
-> fcx_<bench>.csv with FCX columns (fcx_diff etc.) and per-source fc_diff columns for diagnostics."""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "fc")); import importlib.util
spec = importlib.util.spec_from_file_location("ff", str(HERE.parent / "fc/feats.py")); ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)
FC = ff.FC
src = {"S1": json.load(open(HERE.parent / "fc/fc_raw.json"))}
if (HERE / "son_raw.json").exists(): src["S2"] = json.load(open(HERE / "son_raw.json"))
if (HERE / "gem_raw.json").exists():
    g = json.load(open(HERE / "gem_raw.json")); src["G1"] = g["G1"]; src["G2"] = g["G2"]
def bench(c): return "sopbench" if c.count("/") == 2 else ("tauk_banking" if c.startswith("banking") else "tau2")
frames = {}
for s, raw in src.items():
    df = pd.DataFrame([{"case_id": k, **ff.feats(v)} for k, v in raw.items()]).set_index("case_id").reindex(columns=FC)
    frames[s] = df
print("sources", list(frames))
for b in ["sopbench", "tau2", "tauk_banking"]:
    zs = []
    out = pd.DataFrame(index=[c for c in frames["S1"].index if bench(c) == b])
    for s, df in frames.items():
        d = df.reindex(out.index); d = d.fillna(d.median())
        z = (d - d.mean()) / d.std().replace(0, 1); zs.append(z)
        out[f"{s}_diff"] = d["fc_diff"]
    m = sum(zs) / len(zs)
    for c in FC: out[c.replace("fc_", "fcx_")] = m[c]
    out.index.name = "case_id"; out.to_csv(HERE / f"fcx_{b}.csv"); print(b, out.shape)
