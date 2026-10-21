import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, hashlib, threading, re, time, os
from pathlib import Path
REPO = Path(ZR.REPO)
sys.path.insert(0, str(REPO / "src"))
import llm, urllib.request, urllib.error
import pandas as pd, numpy as np
HERE = Path(__file__).parent
CACHE = HERE / "cache.jsonl"
_lock = threading.Lock(); _calls = {"n": 0}
def _load():
    d = {}
    if CACHE.exists():
        for l in CACHE.read_text().splitlines():
            try: r = json.loads(l); d[r["key"]] = r
            except Exception: pass
    return d
_C = _load()
def key(prompt, temp, rep): return hashlib.sha256(f"{temp}|{rep}|{prompt}".encode()).hexdigest()
def gen(prompt, temp=0.0, rep=0, max_tokens=4000, thinking="low"):
    """cached Gemini call with temperature; counts real calls"""
    k = key(prompt, temp, rep)
    if k in _C and _C[k].get("text"): return _C[k]["text"]
    url, auth = llm._endpoint(llm.MODEL)
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temp, "maxOutputTokens": max_tokens, "thinkingConfig": {"thinkingLevel": thinking}}}
    data = json.dumps(body).encode(); text = ""
    for attempt in range(6):
        url, auth = llm._endpoint(llm.MODEL)
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth})
        with _lock: _calls["n"] += 1
        try:
            with urllib.request.urlopen(req, timeout=180) as r: d = json.loads(r.read())
            cand = (d.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")); break
        except urllib.error.HTTPError as e:
            if e.code == 401: llm._TOKEN["exp"] = 0.0
            if e.code in (401, 429, 500, 502, 503, 504) and attempt < 5: time.sleep(min(60, 2 ** attempt + 1)); continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < 5: time.sleep(min(60, 2 ** attempt + 1)); continue
            raise
    with _lock:
        _C[k] = {"key": k, "text": text}
        with open(CACHE, "a") as f: f.write(json.dumps({"key": k, "text": text}) + "\n")
    return text
def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None
def statements(bench):
    return {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(REPO / "data" / bench / "statements.jsonl")}
def compact_rubrics():
    sys.path.insert(0, str(REPO / "third_party/ADeLe-AIEvaluation/src"))
    from adele.rubrics.catalog import RubricsCatalog
    c = RubricsCatalog(); out = {}
    for d in c.acronyms:
        r = c[d]; lines = [f"### {d}: {r.full_name}"]
        for p in re.split(r"\n\s*\n", r.content):
            if p.startswith("The following") and d.startswith("KN"):
                lines.append(p.split("This reflects")[0].strip()[:600])
            if p.startswith("Level"):
                t = p.split("\n")[0]; t = t.split("Example")[0].strip(); lines.append(t)
        out[d] = "\n".join(lines)
    return out
DEMANDS = ['AS','AT','CEc','CEe','CL','KNa','KNc','KNf','KNn','KNs','MCr','MCt','MCu','MS','QLl','QLq','SNs','VO']
