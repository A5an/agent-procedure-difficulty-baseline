"""Re-plan tau2 tasks with the REAL agent tool list (L1: policy + tool list + scenario, no gold). 3 plans per distinct prompt:
t0 (temperature 0) and two samples at temperature 1. gemini-3.8-flash, thinking medium."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, hashlib, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(ZR.REPO)
sys.path.insert(0, str(REPO / "src"))
import llm
HERE = Path(__file__).parent
POL = Path(ZR.DATA + "/tau2-domains/tasks")
CACHE = HERE / "llm_cache.jsonl"
lock = threading.Lock(); cache = {}
if CACHE.exists():
    for l in CACHE.read_text().splitlines():
        d = json.loads(l); cache[d["k"]] = d["r"]
ncalls = 0
TOOLS = json.load(open(HERE / "tool_text.json"))

def call(prompt, temp, tag):
    global ncalls
    k = hashlib.sha256(f"{temp}|{tag}|{prompt}".encode()).hexdigest()
    if k in cache: return cache[k]
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temp, "maxOutputTokens": 8000, "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingLevel": "medium"}}}
    data = json.dumps(body).encode(); text = None
    for attempt in range(8):
        url, auth = llm._endpoint(llm.MODEL)
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth})
        try:
            with urllib.request.urlopen(req, timeout=240) as r: d = json.loads(r.read())
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

INSTR = """You are planning, on paper, how a customer-service agent should handle one customer case. You have no database access and you do not call anything now. Use only the domain policy, the agent's tool list and the customer scenario below.

{policy}

=== TOOLS THE AGENT CAN CALL (name, description, JSON parameter schema) ===
{tools}
=== END TOOLS ===

=== CUSTOMER CASE ===
{case}
=== END CASE ===

Write the plan the agent is expected to follow if it handles the case correctly, assuming the customer data is whatever the scenario implies. Every tool name in the plan must be exactly one of the tool names listed above. If the correct handling needs the customer to do something on their own device or account (the agent cannot do it with its tools), add a step with the tool name "ask_customer_to_act" and put the action in the purpose. Return ONLY JSON with this schema:
{{
 "plan": [ {{"tool": "tool_name_from_the_list", "purpose": "short purpose", "writes_db": true or false}} ],   // ordered tool calls the agent should make (reads and writes), no talking steps
 "policy_rules": ["short text of each policy rule or condition the plan relies on"],
 "facts_to_ask_customer": ["facts the agent must still ask the customer for, not given in the scenario"],
 "outcome": "complete" | "refuse" | "transfer_to_human",   // correct final outcome of the case
 "ambiguity": 0 or 1 or 2   // 0 clear, 1 some missing or unclear information, 2 the right action depends on information that cannot be known from the text
}}"""

def policy(domain):
    if domain == "telecom":
        t = (POL / "telecom_main_policy.md").read_text() + "\n\n" + (POL / "telecom_tech_support_manual.md").read_text()
    else: t = (POL / f"{domain}_policy.md").read_text()
    return f"=== DOMAIN POLICY ({domain}) ===\n{t}\n=== END POLICY ==="

def build_all():
    items = []
    for l in open(REPO / "data/tau2/statements.jsonl"):
        s = json.loads(l); dom = s["case_id"].split("/")[0]
        items.append((s["case_id"], INSTR.format(policy=policy(dom), tools=TOOLS[dom], case=s["text"])))
    return items

def run(jobs, workers=8):
    with ThreadPoolExecutor(workers) as ex: return list(ex.map(lambda j: call(*j), jobs))

if __name__ == "__main__":
    mode = sys.argv[1]; items = build_all()
    uniq = {}
    for c, p in items: uniq.setdefault(p, []).append(c)
    prompts = list(uniq)
    print("tasks", len(items), "distinct prompts", len(prompts), "calls for 3 plans", 3 * len(prompts))
    if mode == "test":
        sel = [prompts[0], prompts[-1]]
        for p, o in zip(sel, run([(p, 0, "t0") for p in sel])): print(o[:1500]); print("----")
    else:
        res = {}
        for tag, temp in (("t0", 0), ("s1", 1.0), ("s2", 1.0)):
            out = run([(p, temp, tag) for p in prompts])
            for p, o in zip(prompts, out):
                for c in uniq[p]: res.setdefault(c, {})[tag] = o
            print(tag, "new calls so far", ncalls, flush=True)
        json.dump(res, open(HERE / "plans3.json", "w"))
        print("missing", sum(v is None for d in res.values() for v in d.values()))
