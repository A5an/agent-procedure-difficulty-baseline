"""FCT: k-means (K=12, seed 0) over bge-small embeddings of all S1 failure-mode sentences; fct_k = sum of p in cluster k."""
import json, sys, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
raw = json.load(open("../fc/fc_raw.json"))
def pj(t):
    t = (t or "").strip()
    i, j = t.find("{"), t.rfind("}")
    try: return json.loads(t[i:j + 1])
    except Exception: return None
rows = []
for cid, t in raw.items():
    d = pj(t) or {}
    for m in d.get("failure_modes", []) or []:
        try: rows.append((cid, str(m["what"]), float(m["p"])))
        except Exception: pass
df = pd.DataFrame(rows, columns=["case_id", "what", "p"])
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
E = model.encode(df.what.tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
km = KMeans(n_clusters=12, random_state=0, n_init=10).fit(E); df["k"] = km.labels_
F = df.pivot_table(index="case_id", columns="k", values="p", aggfunc="sum", fill_value=0.0)
F.columns = [f"fct_{c}" for c in F.columns]
F = F.reindex(list(raw)).fillna(0.0); F.index.name = "case_id"
bench = F.index.to_series().map(lambda c: "sopbench" if c.count("/") == 2 else ("tauk_banking" if c.startswith("banking") else "tau2"))
for b in ["sopbench", "tau2", "tauk_banking"]: F[bench == b].to_csv(f"fct_{b}.csv")
with open("clusters.txt", "w") as f:
    for k in range(12):
        s = df[df.k == k]; d = np.linalg.norm(E[s.index] - km.cluster_centers_[k], axis=1)
        f.write(f"cluster {k}: n={len(s)}, mean p {s.p.mean():.2f}, benches {s.case_id.map(lambda c: 'sop' if c.count('/')==2 else c.split('/')[0]).value_counts().to_dict()}\n")
        for w in s.iloc[np.argsort(d)[:4]].what: f.write("   - " + w[:160] + "\n")
print(open("clusters.txt").read())
