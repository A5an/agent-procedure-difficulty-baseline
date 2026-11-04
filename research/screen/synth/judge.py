"""Independent check of the policy text of every accepted spec by a model of another family (PREREG.md).

  python judge.py <generator> [--fix]

gemini-2.5-flash (through the proxy on 8094) reads the formal rules and the policy text and reports, for every rule,
whether an agent that reads only the policy would apply it exactly (thresholds, direction of comparisons, the outcome
and its write calls), plus any statement in the policy that adds or contradicts a rule. With --fix, specs with a
problem go back to their generator once to rewrite policy_text only, and are judged again. Results in
judge/<generator>.jsonl; fixed policy texts are written into gens/<generator>/<procedure>.json (old text kept in
_policy_before_fix).
"""
import json, sys, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
import dataset, engine, gen

HERE = Path(__file__).resolve().parent
PROXY = "http://127.0.0.1:8094/v1/chat/completions"
MODEL = "check|gemini-3.8-flash"

ASK = """You check a policy document written for customer-service agents against the formal rules a program uses to grade them.

How the program works, so you do not flag its normal behaviour:
- Rules are ordered; the first rule whose condition holds decides the outcome; if none holds, the default outcome applies.
- req.* = what the customer asks for, claim.* = what the customer states, other names = database records.
- claim.* is None when the customer does not state it. "!=" with None is true and ordering comparisons with None are
  false. So "if the caller cannot give X, do not proceed" matches a rule "claim.x != record.x -> refuse".
- Before the rules apply, the program checks whether the agent can identify the records and get what the customer asks
  for. If not, the correct result is to change nothing. Policy statements saying so ("if you cannot find the account,
  make no change") are consistent with the rules and are not conflicts.
- If the customer changes their mind, the rules apply to the request they finally make. Saying so is consistent.
- Every req.* value must come from the customer; if the customer cannot give one, the correct result is to change
  nothing. So "ask the customer for X if they have not said it" is consistent.
- Only decisions and database actions are graded. Extra instructions about what to tell the customer are not conflicts.
- Today is fixed at {today}. A fixed calendar date that is equivalent to a relative condition is fine.
- General good conduct (confirm before acting, do not change what was not asked, verify identity as the rules say) is fine.

Formal rules:
{rules}
Default outcome: {default}

Outcomes and the exact actions each one takes:
{outcomes}

Tools the agent has:
{tools}

Policy document the agent reads:
<<<
{policy}
>>>

For every rule, decide whether a careful agent that reads only the policy document would decide every case the way the
formal rule does (same condition, thresholds, direction and inclusiveness of comparisons, precedence over other rules,
same outcome with the same actions and argument values):
- "exact": yes, possibly with different wording;
- "ambiguous": a careful agent could reasonably decide some concrete case either way;
- "different": the policy states it differently (another threshold, direction, outcome or action);
- "missing": the policy does not state it.
Then list statements in the policy that would make a careful agent decide a concrete case differently from the formal
rules (name the situation and both decisions). Do not list a statement that leads to the same result as the program,
stylistic issues, or far-fetched readings.

Answer with JSON only:
{{"rules": [{{"id": "R1", "verdict": "exact" | "ambiguous" | "different" | "missing", "note": "short reason"}}],
  "conflicts": [{{"situation": "...", "policy_leads_to": "...", "rules_give": "...", "same_result": true | false}}]}}"""


def describe(spec):
    rules = "\n".join(f"- {r['id']}: if {r['when']} -> outcome {r['outcome']}  ({r['explain']})" for r in spec["rules"])
    outs = "\n".join(f"- {n} ({o['kind']}): " + ("; ".join(f"{c['tool']}({', '.join(f'{a}={x}' for a, x in c['args'].items())})"
                                                            for c in o["calls"]) or "no action") + f"  [{o.get('description', '')}]"
                     for n, o in spec["outcomes"].items())
    tools = "\n".join(f"- {t['name']} ({t['kind']} {t['entity']}): {t['description']}" for t in spec["tools"])
    return ASK.format(today=engine.TODAY, rules=rules, default=spec["default_outcome"], outcomes=outs, tools=tools, policy=spec["policy_text"])


def judge(spec):
    r = requests.post(PROXY, json={"model": MODEL, "messages": [{"role": "user", "content": describe(spec)}],
                                   "temperature": 0.0, "max_tokens": 8192, "reasoning_effort": "low"}, timeout=600)
    txt = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    out = json.loads(m.group(0))
    bad = [x for x in out["rules"] if x.get("verdict") != "exact"]
    out["conflicts"] = [c for c in out.get("conflicts", []) if isinstance(c, dict) and c.get("same_result") is False]
    ids = {x.get("id") for x in out["rules"]}
    missing_ids = [r["id"] for r in spec["rules"] if r["id"] not in ids]
    return {"verdicts": out["rules"], "conflicts": out.get("conflicts", []), "ok": not bad and not out.get("conflicts") and not missing_ids,
            "problems": [f"rule {x['id']}: {x['verdict']} ({x.get('note', '')})" for x in bad]
                        + [f"policy statement: in {c.get('situation')}: the policy leads to {c.get('policy_leads_to')}, the rules give {c.get('rules_give')}" for c in out["conflicts"]]
                        + [f"rule {i}: not assessed" for i in missing_ids]}


FIX = """{fmt}

## Your spec

{spec}

## An independent reviewer compared your policy_text with your rules and found these problems

{problems}

Rewrite policy_text only, so that every rule is stated exactly (conditions, numbers, precedence, outcome and its
actions) and nothing in the text adds to or contradicts the rules. Keep the organization's voice. Output only a JSON
object of the form {{"policy_text": "..."}}."""


def one(g, pid, raw, fix):
    spec = json.loads(json.dumps(raw))
    res = {"gen": g, "pid": pid, "round": 0, **judge(spec)}
    log = [res]
    if fix and not res["ok"]:
        clean = {k: v for k, v in raw.items() if not k.startswith("_")}
        txt, meta = gen.call(g, FIX.format(fmt=gen.FMT, spec=json.dumps(clean, indent=1), problems="\n".join(f"- {p}" for p in res["problems"])))
        try:
            new = gen.parse(txt)["policy_text"]
            if len(new) < 300:
                raise ValueError("too short")
            raw["_policy_before_fix"] = raw["policy_text"]
            raw["policy_text"] = new
            json.dump(raw, open(HERE / dataset.GEN_DIR / g / f"{pid}.json", "w"), indent=1)
            res2 = {"gen": g, "pid": pid, "round": 1, "fix_meta": meta, **judge(json.loads(json.dumps(raw)))}
        except Exception as e:
            res2 = {"gen": g, "pid": pid, "round": 1, "fix_meta": meta, "ok": False, "problems": [f"fix failed: {e}"]}
        log.append(res2)
    return log


if __name__ == "__main__":
    g = sys.argv[1]
    fix = "--fix" in sys.argv
    specs = dataset.load_specs(g)
    (HERE / "judge").mkdir(exist_ok=True)
    with ThreadPoolExecutor(4) as ex, open(HERE / "judge" / f"{dataset.V + '_' if dataset.V else ''}{g}.jsonl", "a") as f:
        for log in ex.map(lambda kv: one(g, kv[0], kv[1], fix), specs.items()):
            for r in log:
                f.write(json.dumps(r) + "\n")
                print(g, r["pid"], r["round"], r["ok"], r["problems"][:3], flush=True)
