"""L1: one gemini-3.8-flash call per distinct case writes a Python handler (never executed). usage: gen.py test|tau2|sop"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, hashlib, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(ZR.REPO)
sys.path.insert(0, str(REPO / "src"))
import llm
S = Path(ZR.SCREEN)
HERE = Path(__file__).parent; CACHE = HERE / "llm_cache.jsonl"
POL = Path(ZR.DATA + "/tau2-domains/tasks")
TOOLS = json.load(open(S / "ideas3/plan_tools/tool_text.json"))
DT = json.load(open(S / "ideas3/paper_pilot/domain_tools.json"))
lock = threading.Lock(); cache = {}; ncalls = 0
if CACHE.exists():
    for l in CACHE.read_text().splitlines():
        d = json.loads(l); cache[d["k"]] = d["r"]

def call(prompt):
    global ncalls
    k = hashlib.sha256(prompt.encode()).hexdigest()
    if k in cache: return cache[k]
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 6000, "thinkingConfig": {"thinkingLevel": "low"}}}
    data = json.dumps(body).encode(); text = None
    for attempt in range(8):
        url, auth = llm._endpoint(llm.MODEL)
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth})
        try:
            with urllib.request.urlopen(req, timeout=240) as r: d = json.loads(r.read())
            parts = ((d.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")); break
        except urllib.error.HTTPError as e:
            if e.code == 401: llm._TOKEN["exp"] = 0.0
            if e.code in (401, 429, 500, 502, 503, 504): time.sleep(min(60, 2 ** attempt + 1)); continue
            if e.code == 400:
                print('HTTP400', e.read()[:300], flush=True); break
            raise
        except (urllib.error.URLError, TimeoutError, OSError): time.sleep(min(60, 2 ** attempt + 1))
    with lock:
        ncalls += 1
        if text is not None:
            cache[k] = text
            with open(CACHE, "a") as f: f.write(json.dumps({"k": k, "r": text}) + "\n")
    return text

INSTR = """You are a customer-service agent. Right now you have no database access and nothing will be executed. Write, on paper, a short Python function handle(tools) that would handle THIS customer's case correctly under the policy, using only the real tools listed below. Call tools as tools.tool_name(arg=value). Read tool results and branch on them with if/else exactly where the policy requires a check (eligibility, dates, status, limits, fees), loop over items if the policy needs it, and refuse or transfer (return "refuse" or "transfer") where the policy says so. Use datetime where dates matter. Assume the customer data is whatever the scenario implies. Do not invent tools. Keep it compact: no comments, no prints, no helper functions.

{policy}

=== TOOLS ===
{tools}
=== END TOOLS ===

=== CUSTOMER CASE ===
{case}
=== END CASE ===

Return ONLY the Python source of handle(tools), nothing else."""

def tau_policy(dom):
    t = (POL / "telecom_main_policy.md").read_text() + "\n\n" + (POL / "telecom_tech_support_manual.md").read_text() if dom == "telecom" else (POL / f"{dom}_policy.md").read_text()
    return f"=== DOMAIN POLICY ({dom}) ===\n{t}\n=== END POLICY ==="
def sop_tools(dom):
    L = []
    for tl in DT[dom]["tools"]:
        f = tl["function"]; L.append(f"- {f['name']}({', '.join(f['parameters'].get('properties', {}))}): {f['description']}")
    return "\n".join(L)

def items(which):
    out = []
    if which == "tau2":
        for l in open(REPO / "data/tau2/statements.jsonl"):
            s = json.loads(l); dom = s["case_id"].split("/")[0]
            out.append((s["case_id"], INSTR.format(policy=tau_policy(dom), tools=TOOLS[dom], case=s["text"])))
    else:
        for l in open(REPO / "data/sopbench/statements.jsonl"):
            s = json.loads(l); dom = s["case_id"].split("/")[0]
            st = s["text"].split("\nAvailable tools:")[0]
            out.append((s["case_id"], INSTR.format(policy="(The domain, the policy of the requested action and the customer message are in the case text.)", tools=sop_tools(dom), case=st)))
    return out

if __name__ == "__main__":
    which = sys.argv[1]; its = items("tau2" if which == "test" else which)
    uniq = {}
    for c, p in its: uniq.setdefault(p, []).append(c)
    ps = list(uniq)
    print(which, "cases", len(its), "distinct", len(ps), flush=True)
    if which == "test": ps = [ps[0], ps[-1]]
    with ThreadPoolExecutor(4) as ex: outs = list(ex.map(call, ps))
    if which == "test":
        for o in outs: print(o); print("-----")
    else:
        res = {c: o for p, o in zip(ps, outs) for c in uniq[p]}
        json.dump(res, open(HERE / f"code_{which}.json", "w"))
        print("new calls", ncalls, "missing", sum(v is None for v in res.values()))
