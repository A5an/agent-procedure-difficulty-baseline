"""E5: bge-base-en-v1.5 embeddings of the 50 TauIndianBank families from the text a company has before launch: the
family's purpose line. TauIndianBank's policy.md is one short global text with no per-procedure section, so nothing else
is procedure-specific. No family name, prefix or other benchmark metadata. Recipe as src/embed_features.py of the
baseline repo (normalize_embeddings=True, cosine).  <repo venv python> embed_tib.py -> emb_tib.npz"""
import json, sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import sentence_transformers, torch
HERE = Path(__file__).resolve().parent; SCR = HERE.parent
s1 = json.load(open(SCR / "tib" / "selection.json")); s2 = json.load(open(SCR / "combo_rep" / "selection_rep.json"))
F = s1["test_families"] + s2["test_families"]; P = {**s1["purpose"], **s2["purpose"]}
texts = [P[f] for f in F]
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
X = m.encode(texts, batch_size=16, normalize_embeddings=True)
np.savez_compressed(HERE / "emb_tib.npz", families=np.array(F, dtype=object), X=X, texts=np.array(texts, dtype=object))
print("emb", X.shape, "versions: sentence-transformers", sentence_transformers.__version__, "torch", torch.__version__, "numpy", np.__version__)
