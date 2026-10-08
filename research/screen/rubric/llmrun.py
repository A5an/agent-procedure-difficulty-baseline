"""Cached LLM runner for the three rubrics. Run: python llmrun.py <rubric|all> [--agree]"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, os, sys, threading, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(ZR.REPO)
sys.path.insert(0, str(REPO / "src"))
import subprocess
os.environ["LLM_BACKEND"] = "vertex"
os.environ["GOOGLE_CLOUD_PROJECT"] = subprocess.run(["gcloud", "config", "get-value", "project"],
                                                    capture_output=True, text=True).stdout.strip()
import llm  # noqa
import prompts as P

HERE = Path(__file__).parent
CACHE = HERE / "cache"
BENCHES = ["sopbench", "tau2", "tauk_banking"]
LOCK = threading.Lock()
MAXCALLS = 9000
calls = {"n": 0}

def statements(bench):
    return [json.loads(l) for l in open(REPO / "data" / bench / "statements.jsonl")]

def validate(rubric, d):
    if not isinstance(d, dict):
        return False
    if rubric == "proc":
        st = d.get("steps")
        if not isinstance(st, list) or not (1 <= len(st) <= 60):
            return False
        for s in st:
            if not isinstance(s, dict) or s.get("type") not in P.STEP_TYPES or not isinstance(s.get("depth"), int):
                return False
        return (isinstance(d.get("refusal_conditions"), int) and d.get("needs_hidden_data") in (0, 1)
                and isinstance(d.get("p_refusal"), (int, float)) and 0 <= d["p_refusal"] <= 1)
    if rubric == "lara":
        return all(isinstance(d.get(f"D{i}"), int) and 1 <= d[f"D{i}"] <= 5 for i in range(1, 6))
    if rubric == "rpa":
        return all(isinstance(d.get(n), int) and 0 <= d[n] <= 2 for n, _, _ in P.RPA_CRITERIA)

def parse(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    a, b = t.find("{"), t.rfind("}")
    return json.loads(t[a:b + 1])

def call(rubric, rep, case_id, text):
    prompt = {"proc": P.proc_prompt, "lara": P.lara_prompt, "rpa": P.rpa_prompt}[rubric](text)
    last = ""
    for attempt in range(4):
        p = prompt if attempt == 0 else prompt + "\n\nYour previous answer was not valid. Return exactly one valid JSON object in the requested form, nothing else."
        with LOCK:
            if calls["n"] >= MAXCALLS:
                raise SystemExit("call budget reached")
            calls["n"] += 1
        try:
            out, _ = llm.generate(p, max_tokens=6000)
            last = out
            d = parse(out)
            if validate(rubric, d):
                return {"key": f"{rubric}|{rep}|{case_id}", "parsed": d, "attempts": attempt + 1, "ok": True}
        except SystemExit:
            raise
        except Exception as e:
            last = f"ERR {type(e).__name__}"
    return {"key": f"{rubric}|{rep}|{case_id}", "parsed": None, "raw": last[:500], "attempts": 4, "ok": False}

def sample100():
    allt = [(b, r["case_id"]) for b in BENCHES for r in statements(b)]
    random.Random(0).shuffle(allt)
    return allt[:100]

def run(rubric, rep=1, only=None):
    path = CACHE / f"{rubric}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["key"] for l in open(path) if json.loads(l).get("ok")}
    todo = []
    for b in BENCHES:
        for r in statements(b):
            if only is not None and (b, r["case_id"]) not in only:
                continue
            if f"{rubric}|{rep}|{r['case_id']}" not in done:
                todo.append((r["case_id"], r["text"]))
    print(rubric, "rep", rep, "to do", len(todo), flush=True)
    n = 0
    with ThreadPoolExecutor(24) as ex, open(path, "a") as f:
        for rec in ex.map(lambda x: call(rubric, rep, *x), todo):
            f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1
            if n % 200 == 0:
                print(n, "done, calls", calls["n"], flush=True)
    bad = sum(1 for _ in [1])  # placeholder
    print(rubric, "finished; calls this process", calls["n"])

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "test":
        recs = []
        for b in BENCHES:
            r = statements(b)[3]
            x = {rb: call(rb, 0, r["case_id"], r["text"]) for rb in ("proc", "lara", "rpa")}
            print(b, json.dumps({k: v["parsed"] for k, v in x.items()})[:1500])
        sys.exit()
    for rb in (["proc", "lara", "rpa"] if which == "all" else [which]):
        run(rb)
    if "--agree" in sys.argv:
        s = set(sample100())
        run("proc", rep=2, only=s)
