"""G6 label audit (PREREG.md): why did sampled failed episodes fail? gemini-3.8-flash (low reasoning, as judge.py) reads spec, case and conversation.

  python audit.py   -> audit.jsonl (20 failed episodes per generator, pooled over agents, seed 20261105)
"""
import json, random, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
import dataset, engine

HERE = Path(__file__).resolve().parent
PROXY = "http://127.0.0.1:8094/v1/chat/completions"
GENS, AGENTS = ["haiku", "sonnet", "opus"], ["lite", "flash", "g38"]

ASK = """A customer-service AI agent was graded as FAILED on a test case. Decide why.

The grader compares the final database with the database produced by the correct actions below. The correct outcome
was computed by a program from formal rules; the agent only saw the policy document and the tools.

Policy document the agent saw:
<<<
{policy}
>>>

Formal rules used by the grader (first match wins, else default "{default}"):
{rules}

What the simulated customer was told (their scenario):
<<<
{scenario}
>>>

Correct outcome: {outcome} (decided by {decided_by}). Correct write actions: {calls}

Conversation (agent tool calls and tool results included):
<<<
{conv}
>>>

Write actions the agent actually made: {agent_writes}

Classify the main cause of the failure:
- "agent": the correct outcome follows from the policy and from what happened in the conversation, and the agent got it
  wrong (wrong, missing or extra action, wrong argument, skipped a check, claimed an action it did not make, gave up).
- "label": the correct outcome is NOT justified by the policy as written and the facts available in the conversation
  (the policy is unclear, omits or contradicts the rule, the case data are inconsistent), so a careful agent could fail.
- "user": the simulated customer broke its scenario (invented or withheld facts it should give, refused a confirmation
  it should give, ended too early, asked for something else) so success became impossible or unreasonable.
- "unclear": cannot tell.

Answer with JSON only: {{"cause": "agent" | "label" | "user" | "unclear", "detail": "one sentence"}}"""


def conv(msgs):
    out = []
    for m in msgs:
        r = m["role"]
        if r == "assistant":
            if m.get("content"):
                out.append(f"AGENT: {m['content']}")
            for tc in m.get("tool_calls") or []:
                out.append(f"AGENT CALLS {tc['name']}({json.dumps(tc.get('arguments'), default=str)})")
        elif r == "tool":
            out.append(f"TOOL RESULT: {str(m.get('content'))[:1200]}")
        elif r == "user":
            out.append(f"CUSTOMER: {m.get('content')}")
    return "\n".join(out)


def item(it, agent):
    pre = f"{dataset.V}_" if dataset.V else ""
    msgs = json.load(open(HERE / "runs" / "msgs" / f"{pre}{it['gen']}_{agent}" / f"{it['uid']}.json"))
    spec = it["spec"]
    tools = {t["name"]: t for t in spec["tools"]}
    writes = [f"{tc['name']}({json.dumps(tc.get('arguments'), default=str)})" for m in msgs if m["role"] == "assistant"
              for tc in m.get("tool_calls") or [] if tc["name"] in tools and engine.tool_is_write(tools[tc["name"]])]
    us = it["task"]["user_scenario"]
    scen = f"Persona: {us['persona']}\n" + "\n".join(f"{k}: {v}" for k, v in us["instructions"].items() if v)
    return ASK.format(policy=f"Today is {engine.TODAY}.\n" + spec["policy_text"], default=spec["default_outcome"],
                      rules="\n".join(f"- {r['id']}: if {r['when']} -> {r['outcome']} ({r['explain']})" for r in spec["rules"]),
                      scenario=scen, outcome=it["sol"]["outcome"], decided_by=it["sol"]["decided_by"],
                      calls=json.dumps(it["sol"]["calls"], default=str) or "none", conv=conv(msgs), agent_writes="; ".join(writes) or "none")


def ask(prompt):
    r = requests.post(PROXY, json={"model": "check|gemini-3.8-flash", "messages": [{"role": "user", "content": prompt}],
                                   "temperature": 0.0, "max_tokens": 8192, "reasoning_effort": "low"}, timeout=600)
    txt = r.json()["choices"][0]["message"]["content"]
    return json.loads(re.search(r"\{.*\}", txt, re.S).group(0))


if __name__ == "__main__":
    rng = random.Random(20261105)
    jobs = []
    for g in GENS:
        items = {it["uid"]: it for it in dataset.cases(g)}
        fails = []
        for a in AGENTS:
            p = HERE / "runs" / f"{g}_{a}.jsonl"
            if p.exists():
                fails += [(json.loads(l)["task_id"], a) for l in open(p) if json.loads(l).get("reward") == 0.0]
        for uid, a in rng.sample(sorted(fails), min(20, len(fails))):
            jobs.append((g, a, items[uid]))
    def run(j):
        g, a, it = j
        try:
            res = ask(item(it, a))
        except Exception as e:
            res = {"cause": "error", "detail": str(e)[:200]}
        return {"gen": g, "agent": a, "uid": it["uid"], "level": it["plan"]["level"], "behavior": it["plan"]["behavior"], **res}
    with ThreadPoolExecutor(4) as ex, open(HERE / "audit.jsonl", "w") as f:
        for r in ex.map(run, jobs):
            f.write(json.dumps(r) + "\n")
            print(r["gen"], r["agent"], r["uid"], r["cause"], r["detail"][:120], flush=True)
