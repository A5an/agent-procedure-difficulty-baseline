"""Prior conflict (L1): what a policy-blind assistant would do vs what the policy requires.
Per case: 6 naive plans without the policy (gemini-3.8-flash t0 + 2 samples at t=1; gemini-2.5-flash-lite t0 + 2 at t=1),
1 policy plan with the policy (flash t0, also rates how expected each rule is), 1 matching call (flash t0) that says which
policy rules each naive plan covers. usage: prior.py test | run"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "common")); sys.path.insert(0, ZR.SCREEN + "/ideas9/gap")
import vx
from gap import inputs  # (bench, case_id, policy_with_header, tools, request)
C = vx.Client(HERE)
NAIVE = [("gemini-3.8-flash", 0.0, 0), ("gemini-3.8-flash", 1.0, 1), ("gemini-3.8-flash", 1.0, 2),
         ("gemini-2.5-flash-lite", 0.0, 0), ("gemini-2.5-flash-lite", 1.0, 1), ("gemini-2.5-flash-lite", 1.0, 2)]

def role(bench, case_id, pol):
    if bench == "sopbench": return pol.split("\n\nPolicy for the requested action:")[0].strip()
    dom = case_id.split("/")[0]
    if bench == "tau2": return f"Domain: {dom} customer service. You are a customer service agent of a {dom} company and you help customers using the tools below."
    return "Domain: banking customer service. You are a customer service agent of a bank and you help customers using the tools below."

P_NAIVE = """{role}

No company policy or rulebook is given to you. Handle the customer the way a careful, competent assistant would by default, using only common sense and the tools.

=== TOOLS ===
{tools}
=== END TOOLS ===

=== CUSTOMER REQUEST OR SCENARIO ===
{request}
=== END ===

On paper (you call nothing), write what you would do. Return ONLY JSON:
{{"calls": [{{"tool": "tool name", "kind": "check" or "lookup" or "action"}}],
  "checks": ["one short sentence for each condition you would verify before acting"],
  "decision": "perform" or "refuse" or "depends_on_checks" or "ask_more" or "transfer"}}"""

P_POLICY = """{role}

You must follow the company policy below exactly.

=== POLICY ===
{policy}
=== END POLICY ===

=== TOOLS ===
{tools}
=== END TOOLS ===

=== CUSTOMER REQUEST OR SCENARIO ===
{request}
=== END ===

On paper (you call nothing), write what the policy requires for this case. Return ONLY JSON:
{{"calls": [{{"tool": "tool name", "kind": "check" or "lookup" or "action"}}],
  "rules": [{{"rule": "one short sentence for each policy rule or condition that applies to this case",
             "default": 1 to 5 (5 = any careful assistant would apply this rule even without being told, 1 = nobody would think of it without the policy)}}],
  "decision": "perform" or "refuse" or "depends_on_checks" or "ask_more" or "transfer"}}"""

P_MATCH = """Below are the policy rules that apply to one customer case, and the checks that several assistants who did NOT read the policy said they would verify. For every policy rule, list the numbers of the assistants whose checks or calls cover that rule (the same condition, possibly in other words). Be strict: a vague or different check does not count.

=== POLICY RULES ===
{rules}
=== ASSISTANTS ===
{naive}
=== END ===

Return ONLY JSON: {{"coverage": [{{"rule": rule number, "covered_by": [assistant numbers]}}]}}"""

def naive_prompt(it): return P_NAIVE.format(role=role(it[0], it[1], it[2]), tools=it[3], request=it[4])
def policy_prompt(it):
    pol = it[2].split("Policy for the requested action:\n", 1)[1] if it[0] == "sopbench" else it[2]
    return P_POLICY.format(role=role(it[0], it[1], it[2]), policy=pol, tools=it[3], request=it[4])

def stage1(items, nthreads=int(__import__("os").environ.get("NT", "12"))):
    jobs = []
    for it in items:
        for m, t, s in NAIVE: jobs.append((naive_prompt(it), m, t, s, "naive"))
        jobs.append((policy_prompt(it), "gemini-3.8-flash", 0.0, 0, "policy"))
    uniq = list(dict.fromkeys(jobs)); random.Random(0).shuffle(uniq)
    with ThreadPoolExecutor(nthreads) as ex: list(ex.map(lambda j: C.call(j[0], model=j[1], temp=j[2], seed=j[3], tag=j[4]), uniq))

def get(it):
    nv = [vx.parse_json(C.call(naive_prompt(it), model=m, temp=t, seed=s, tag="naive")) for m, t, s in NAIVE]
    pp = vx.parse_json(C.call(policy_prompt(it), model="gemini-3.8-flash", tag="policy"))
    return nv, pp

def match_prompt(nv, pp):
    rules = "\n".join(f"{i+1}. {r.get('rule','')}" for i, r in enumerate((pp or {}).get("rules", [])))
    nav = []
    for k, n in enumerate(nv):
        if not n: nav.append(f"Assistant {k+1}: (no answer)"); continue
        calls = ", ".join(c.get("tool", "") for c in n.get("calls", []) if isinstance(c, dict))
        chk = "; ".join(str(x) for x in n.get("checks", []))
        nav.append(f"Assistant {k+1}: calls [{calls}]; checks: {chk}")
    return P_MATCH.format(rules=rules or "(none)", naive="\n".join(nav))

def stage2(items, nthreads=int(__import__("os").environ.get("NT", "12"))):
    ps = []
    for it in items:
        nv, pp = get(it)
        if pp and pp.get("rules"): ps.append(match_prompt(nv, pp))
    ps = list(dict.fromkeys(ps))
    with ThreadPoolExecutor(nthreads) as ex: list(ex.map(lambda p: C.call(p, model="gemini-3.8-flash", tag="match"), ps))

def collect(items):
    out = {}
    for it in items:
        nv, pp = get(it); mt = None
        if pp and pp.get("rules"): mt = vx.parse_json(C.call(match_prompt(nv, pp), model="gemini-3.8-flash", tag="match"))
        out[it[1]] = {"bench": it[0], "naive": nv, "policy": pp, "match": mt}
    return out

if __name__ == "__main__":
    items = inputs(); mode = sys.argv[1]
    if mode == "test":
        rnd = random.Random(7)
        items = rnd.sample([i for i in items if i[0] == "sopbench"], 4) + rnd.sample([i for i in items if i[0] == "tau2"], 4) + rnd.sample([i for i in items if i[0] == "banking"], 2)
    stage1(items); print("stage1 calls", C.ncalls, "fails", C.nfail, flush=True)
    stage2(items); print("stage2 calls", C.ncalls, "fails", C.nfail, flush=True)
    res = collect(items)
    json.dump(res, open(HERE / ("test_raw.json" if mode == "test" else "prior_raw.json"), "w"))
    print("done", len(res), "calls", C.ncalls, "fails", C.nfail)
