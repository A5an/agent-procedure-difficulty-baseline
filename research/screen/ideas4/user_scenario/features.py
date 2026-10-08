import json, pandas as pd, numpy as np
from pathlib import Path
HERE = Path(__file__).parent
raw = json.load(open(HERE / "raw.json"))
COLS = ["n_goals", "n_mind_changes", "withholds_info", "policy_forbidden_request", "pressure", "n_conditional", "needs_computation", "ambiguous_reference", "n_constraints"]
def parse(s):
    try: return json.loads(s)
    except Exception: return None
rows, cls, rows2 = [], {}, []
for c, s in raw["scn"].items():
    d = parse(s)
    if d is None: print("bad", c); continue
    rows.append({"case_id": c, **{k: float(d.get(k, 0) or 0) for k in COLS}, "n_clauses": float(len(d.get("clauses", [])))})
    dom = c.split("/")[0]; cls[c] = sorted({f"{dom}/{x}" for x in d.get("clauses", [])})
df = pd.DataFrame(rows).set_index("case_id"); df.to_csv(HERE / "scn_tau2.csv"); json.dump(cls, open(HERE / "clause_map.json", "w"))
print(df.describe().T[["mean", "std", "min", "max"]]); print("cases", len(df))
# stability
r2 = {c: parse(s) for c, s in raw["scn2"].items() if s}
from scipy.stats import spearmanr
common = [c for c in r2 if r2[c] and c in df.index]
print("stability on", len(common), "cases")
for k in COLS + ["n_clauses"]:
    a = df.loc[common, k].to_numpy(); b = np.array([float(r2[c].get(k, 0) or 0) if k != "n_clauses" else len(r2[c].get("clauses", [])) for c in common])
    rho = spearmanr(a, b)[0] if a.std() > 0 and b.std() > 0 else float("nan")
    print(f"{k:26s} spearman {rho:+.2f} exact {np.mean(a == b):.2f}")
def _j(c):
    a = set(cls[c]); b = {c.split("/")[0] + "/" + x for x in r2[c].get("clauses", [])}
    return len(a & b) / max(1, len(a | b))
ja = [_j(c) for c in common]
print("clause Jaccard mean", np.mean(ja))
