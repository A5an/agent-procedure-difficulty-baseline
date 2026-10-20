"""Run the judge with cache. usage: run_judge.py <thinking> <variant: all|sop150> [limit]"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, hashlib, random, threading, time
sys.path.insert(0, ZR.REPO + "/src")
import llm
from concurrent.futures import ThreadPoolExecutor
think = sys.argv[1]; variant = sys.argv[2]; limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
rows = json.load(open(os.environ.get("SAMPLE", "sample.json")))
if variant == "sop150":
    sop = [r for r in rows if r["bench"] == "sopbench"]; random.Random(777).shuffle(sop); rows = sorted(sop[:150], key=lambda r: r["id"])
if limit: rows = rows[:limit]
CACHE = f"cache_{think}.jsonl"; cache = {}
if os.path.exists(CACHE):
    for l in open(CACHE): d = json.loads(l); cache[d["key"]] = d
lock = threading.Lock(); calls = [0]
def one(r):
    key = hashlib.sha256((think + r["prompt"]).encode()).hexdigest()
    if key in cache: return
    try: text, usage = llm.generate(r["prompt"], max_tokens=2000, thinking=think)
    except Exception as e: text, usage = "", {"error": type(e).__name__}
    d = {"key": key, "id": r["id"], "text": text, "tok": usage.get("totalTokenCount"), "fin": usage.get("finishReason")}
    with lock:
        calls[0] += 1; cache[key] = d; open(CACHE, "a").write(json.dumps(d) + "\n")
t = time.time()
with ThreadPoolExecutor(8) as ex: list(ex.map(one, rows))
print("new calls", calls[0], "cached total", len(cache), "sec", round(time.time() - t))
