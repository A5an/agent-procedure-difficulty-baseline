import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, pickle, numpy as np, torch
S = ZR.SCREEN
sys.path.insert(0, S + "/ideas"); sys.path.insert(0, ".")
from cpulock import cpu_lock
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import prep
M = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
tok = AutoTokenizer.from_pretrained(M); mod = AutoModelForSequenceClassification.from_pretrained(M).eval()
print(mod.config.id2label)
torch.set_num_threads(4)
pairs = {}
for c, v in prep.CASES.items():
    for i, cd in enumerate(v["conds"]):
        for j, s in enumerate(v["sents"] + [v["msg"]]):
            pairs[(s, cd["text"])] = None
keys = list(pairs); print(len(keys), flush=True)
keys.sort(key=lambda k: len(k[0]) + len(k[1]))
res = {}
with cpu_lock("case_signals_nli"):
    with torch.no_grad():
        for b in range(0, len(keys), 32):
            kb = keys[b:b + 32]
            x = tok([k[0] for k in kb], [k[1] for k in kb], return_tensors="pt", padding=True, truncation=True, max_length=384)
            p = torch.softmax(mod(**x).logits, -1).numpy()
            for k, row in zip(kb, p): res[k] = row
            if b % 640 == 0: print(b, flush=True)
pickle.dump({"id2label": mod.config.id2label, "res": res}, open("nli_cache.pkl", "wb"))
