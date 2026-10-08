import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, hashlib, threading, time, urllib.request, urllib.error, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(ZR.REPO)
sys.path.insert(0, str(REPO / "src"))
import llm
HERE = Path(__file__).parent
POL = Path(ZR.DATA + "/tau2-domains/tasks")
CACHE = HERE / "llm_cache.jsonl"
lock = threading.Lock(); cache = {}; ncalls = 0
if CACHE.exists():
    for l in CACHE.read_text().splitlines():
        d = json.loads(l); cache[d["k"]] = d["r"]
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
def policy(domain):
    if domain == "telecom": t = (POL / "telecom_main_policy.md").read_text() + "\n\n" + (POL / "telecom_tech_support_manual.md").read_text()
    else: t = (POL / f"{domain}_policy.md").read_text()
    return f"=== DOMAIN POLICY ({domain}) ===\n{t}\n=== END POLICY ==="
CL_INSTR = """{policy}

Split the policy above into its distinct clauses (rules, procedures, constraints, eligibility conditions, troubleshooting steps). Assign each a short stable id of the form C01, C02, ... and a heading of at most 8 words. Aim for 25 to 45 clauses at a granularity where one customer request typically touches 1 to 4 of them. Return ONLY JSON: {{"clauses": [{{"id": "C01", "heading": "..."}}]}}"""
SC_INSTR = """{policy}

=== CLAUSE LIST (ids assigned to the policy above) ===
{clauses}
=== END CLAUSE LIST ===

Below is the script given to a SIMULATED CUSTOMER who will talk to a customer-service agent that follows the policy. Do not solve the case. Describe how the CUSTOMER behaves according to the script, and which policy clauses the case touches.

=== CUSTOMER SCRIPT ===
{case}
=== END SCRIPT ===

Return ONLY JSON with these keys:
{{"n_goals": integer, number of separate goals or requests the customer pursues in the conversation,
 "n_mind_changes": integer, times the customer changes their mind, reverses or abruptly switches topic,
 "withholds_info": 0 or 1, customer does not volunteer needed information until asked or hides a fact,
 "policy_forbidden_request": 0 or 1, customer asks for something the policy forbids or the agent should refuse,
 "pressure": 0, 1 or 2, how much the customer insists, argues, claims prior approval or applies pressure (0 none, 2 strong),
 "n_conditional": integer, number of conditional instructions of the form if the agent says/offers X then do Y,
 "needs_computation": 0 or 1, the agent must compute something such as refund totals or price differences,
 "ambiguous_reference": 0 or 1, customer refers to items vaguely or ambiguously so the agent must disambiguate,
 "n_constraints": integer, number of customer preferences or constraints that restrict acceptable outcomes (budget, dates, do not want X),
 "clauses": list of clause ids (from the list) that the case touches}}"""
def clause_text(domain):
    p = policy(domain)
    out = call(CL_INSTR.format(policy=p), 0, "clauses")
    return p, json.loads(out)["clauses"]
def main():
    texts = {}
    for l in open(REPO / "data/tau2/statements.jsonl"):
        s = json.loads(l); texts.setdefault(s["text"], []).append(s["case_id"])
    doms = ["airline", "retail", "telecom"]
    with ThreadPoolExecutor(3) as ex: res = list(ex.map(clause_text, doms))
    CL = {d: r for d, r in zip(doms, res)}
    json.dump({d: r[1] for d, r in CL.items()}, open(HERE / "clauses.json", "w"), indent=1)
    jobs = []
    for t, cs in texts.items():
        d = cs[0].split("/")[0]; p, cl = CL[d]
        jobs.append((t, SC_INSTR.format(policy=p, clauses="\n".join(f"{c['id']}: {c['heading']}" for c in cl), case=t)))
    print("distinct texts", len(jobs), "clause calls done", ncalls, flush=True)
    with ThreadPoolExecutor(8) as ex: outs = list(ex.map(lambda j: call(j[1], 0, "scn"), jobs))
    # stability pass on 60 texts, temperature 0.7 same prompt
    import random; random.Random(0).shuffle(jobs); sub = jobs[:60]
    with ThreadPoolExecutor(8) as ex: outs2 = list(ex.map(lambda j: call(j[1], 0.7, "scn2"), sub))
    first = dict(zip([j[0] for j in jobs], []))
    res = {}
    for (t, p), o in zip([(j[0], j[1]) for j in sorted(jobs, key=lambda j: j[0])], []): pass
    allp = {t: call(pr, 0, "scn") for t, pr in jobs}
    sec = {t: call(pr, 0.7, "scn2") for t, pr in sub}
    json.dump({"scn": {c: allp[t] for t, cs in texts.items() for c in cs}, "scn2": {c: sec[t] for t, cs in texts.items() if t in sec for c in cs}},
              open(HERE / "raw.json", "w"))
    print("total new calls", ncalls, "missing", sum(v is None for v in allp.values()))
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        texts = {}
        for l in open(REPO / "data/tau2/statements.jsonl"):
            s = json.loads(l); texts.setdefault(s["text"], []).append(s["case_id"])
        t = list(texts)[0]; p, cl = clause_text("airline"); print(cl[:5], len(cl))
        print(call(SC_INSTR.format(policy=p, clauses="\n".join(f"{c['id']}: {c['heading']}" for c in cl), case=t), 0, "scn"))
    else: main()
