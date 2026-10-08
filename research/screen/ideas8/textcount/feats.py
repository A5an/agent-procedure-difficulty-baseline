"""Pure-code request features (L0), same definition on every benchmark. usage: feats.py"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re, math, collections
import pandas as pd
R = ZR.REPO + "/data/"
def request_text(bench, text):
    if bench == "sopbench":
        return re.search(r"Customer message: (.*?)\nAvailable tools", text, re.S).group(1)
    return text.split("Customer scenario:", 1)[-1]   # tau2: the scenario is the request
def ent(s):
    c = collections.Counter(s); n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0
def feats(req):
    toks8 = re.findall(r"[A-Za-z0-9_]{8,}", req)
    return dict(
        dig_chars=sum(ch.isdigit() for ch in req),                     # DIG
        dig_nums=len(re.findall(r"\d[\d,.]*", req)),                   # DIG
        dig_ids=len(re.findall(r"\b\w*[_\d]\w*\b", req)),              # DIG
        cred_maxent=max([ent(t) for t in toks8], default=0.0),         # CRED
        cred_word=int(bool(re.search(r"password|identification|identity|\bid\b", req.lower()))))  # CRED
if __name__ == "__main__":
    for b in ["sopbench", "tau2", "tauk_banking"]:
        rows = []
        for l in open(R + b + "/statements.jsonl"):
            r = json.loads(l); f = feats(request_text(b, r["text"])); f["case_id"] = r["case_id"]; rows.append(f)
        pd.DataFrame(rows).set_index("case_id").to_csv(f"txt_{b}.csv"); print(b, len(rows))
