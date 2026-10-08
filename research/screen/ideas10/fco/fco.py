import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, os, hashlib, subprocess, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "fc")); sys.path.insert(0, ZR.SCREEN + "/ideas9/gap")
import fc as FC1
from gap import inputs
CACHE = HERE / "opus_cache.jsonl"; cache = {}; lock = threading.Lock()
if CACHE.exists():
    for l in CACHE.read_text().splitlines(): d = json.loads(l); cache[d["k"]] = d["r"]
def call(prompt):
    k = hashlib.sha256(prompt.encode()).hexdigest()
    if k in cache: return cache[k]
    cmd = ["claude", "-p", "--model", "opus", "--tools", "", "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
    text = None
    for a in range(5):
        try: res = json.loads(subprocess.run(cmd, input=prompt, cwd=HERE / "work", capture_output=True, text=True, timeout=900).stdout)
        except Exception as e: res = {"is_error": True, "result": str(e)}
        if not res.get("is_error") and res.get("result"): text = res["result"]; break
        time.sleep(30 * (a + 1))
    if text:
        with lock:
            cache[k] = text
            with open(CACHE, "a") as f: f.write(json.dumps({"k": k, "r": text}) + "\n")
    return text
items = [i for i in inputs() if i[0] != "sopbench"]
ps = {}
for it in items: ps.setdefault(FC1.prompt_of(it), []).append(it[1])
keys = list(ps)
with ThreadPoolExecutor(int(os.environ.get("NW", "4"))) as ex: outs = list(ex.map(call, keys))
json.dump({c: o for p, o in zip(keys, outs) for c in ps[p]}, open(HERE / "fco_raw.json", "w")); print("done", sum(o is None for o in outs), "missing")
