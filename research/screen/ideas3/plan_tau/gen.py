"""Draft an action plan per task with gemini-3.8-flash (L1: one call per case, no tools, no environment)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, hashlib, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(ZR.REPO)
sys.path.insert(0, str(REPO / "src"))
import llm
HERE = Path(__file__).parent
POL = Path(ZR.DATA + "/tau2-domains/tasks")
CACHE = HERE / "llm_cache.jsonl"
lock = threading.Lock()
cache = {}
if CACHE.exists():
    for l in CACHE.read_text().splitlines():
        d = json.loads(l); cache[d["k"]] = d["r"]
ncalls = 0

def call(prompt, temp, tag):
    global ncalls
    k = hashlib.sha256(f"{temp}|{tag}|{prompt}".encode()).hexdigest()
    if k in cache: return cache[k]
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temp, "maxOutputTokens": 4000, "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingLevel": "low"}}}
    data = json.dumps(body).encode()
    import urllib.request, urllib.error
    text = None
    for attempt in range(8):
        url, auth = llm._endpoint(llm.MODEL)
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth})
        try:
            with urllib.request.urlopen(req, timeout=180) as r: d = json.loads(r.read())
            cand = (d.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")); break
        except urllib.error.HTTPError as e:
            if e.code == 401: llm._TOKEN["exp"] = 0.0
            if e.code in (401, 429, 500, 502, 503, 504): time.sleep(min(60, 2 ** attempt + 1)); continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError): time.sleep(min(60, 2 ** attempt + 1))
    with lock:
        ncalls += 1
        if text is not None:
            cache[k] = text
            with open(CACHE, "a") as f: f.write(json.dumps({"k": k, "r": text}) + "\n")
    return text

INSTR = """You are planning, on paper, how a customer-service agent should handle one customer case. You have no tools and no database access. Use only the domain policy and the customer scenario below.

{policy}

=== CUSTOMER CASE ===
{case}
=== END CASE ===

Write the plan the agent is expected to follow if it handles the case correctly, assuming the customer data is whatever the scenario implies. Return ONLY JSON with this schema:
{{
 "plan": [ {{"tool": "snake_case_tool_name", "purpose": "short purpose", "writes_db": true or false}} ],   // ordered tool calls the agent should make (reads and writes), no talking steps
 "policy_rules": ["short text of each policy rule or condition the plan relies on"],
 "facts_to_ask_customer": ["facts the agent must still ask the customer for, not given in the scenario"],
 "outcome": "complete" | "refuse" | "transfer_to_human",   // correct final outcome of the case
 "ambiguity": 0 or 1 or 2   // 0 clear, 1 some missing or unclear information, 2 the right action depends on information that cannot be known from the text
}}"""

def tau2_policy(domain):
    if domain == "telecom":
        t = (POL / "telecom_main_policy.md").read_text() + "\n\n" + (POL / "telecom_tech_support_manual.md").read_text()
    else:
        t = (POL / f"{domain}_policy.md").read_text()
    return f"=== DOMAIN POLICY ({domain}) ===\n{t}\n=== END POLICY ==="

def build_all():
    items = []
    for l in open(REPO / "data/tau2/statements.jsonl"):
        s = json.loads(l); dom = s["case_id"].split("/")[0]
        items.append(("tau2", s["case_id"], INSTR.format(policy=tau2_policy(dom), case=s["text"])))
    for l in open(REPO / "data/sopbench/statements.jsonl"):
        s = json.loads(l)
        items.append(("sopbench", s["case_id"], INSTR.format(policy="(The domain description, the policy of the requested action and the available tools are part of the case text.)", case=s["text"])))
    return items

def run(jobs, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: call(*j), jobs))

if __name__ == "__main__":
    mode = sys.argv[1]
    items = build_all()
    if mode == "test":
        sel = [i for i in items if i[0] == "tau2"][:2] + [i for i in items if i[0] == "sopbench"][:2]
        for i, o in zip(sel, run([(i[2], 0, "t0") for i in sel])): print(i[1], o[:600])
    elif mode == "main":
        out = run([(i[2], 0, "t0") for i in items])
        res = {f"{i[0]}|{i[1]}": o for i, o in zip(items, out)}
        json.dump(res, open(HERE / "plans_t0.json", "w"))
        print("new calls", ncalls, "missing", sum(v is None for v in res.values()))
    elif mode == "stab":
        import random
        rnd = random.Random(0)
        tau = [i for i in items if i[0] == "tau2"]; sop = [i for i in items if i[0] == "sopbench"]
        # 100 tasks: 50 tau2 (spread over domains, distinct texts only) + 50 SOPBench
        seen, tsel = set(), []
        for i in rnd.sample(tau, len(tau)):
            if i[2] not in seen: seen.add(i[2]); tsel.append(i)
        tsel = tsel[:50]; ssel = rnd.sample(sop, 50); sel = tsel + ssel
        res = {}
        for rep in (1, 2):
            out = run([(i[2], 1.0, f"s{rep}") for i in sel])
            for i, o in zip(sel, out): res.setdefault(f"{i[0]}|{i[1]}", []).append(o)
        json.dump(res, open(HERE / "plans_t1.json", "w"))
        print("new calls", ncalls)
