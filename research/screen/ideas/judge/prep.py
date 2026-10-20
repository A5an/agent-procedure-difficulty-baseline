"""Build the judge sample and prompts. No LLM calls. Seed fixed (20261005)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re, random, glob, os, collections
import pandas as pd
REPO = ZR.REPO + "/data"
EXT = ZR.DATA
SOP = f"{EXT}/SOPBench-d2622008"; TAU = f"{EXT}/tau2-domains"
DOMAINS = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]
SOP_AGENTS = ["gpt-4.1-mini-mode_fc", "gemini-2.0-flash-001-mode_fc"]
TAU_AGENTS = ["gpt-4-1-mini_openai_2024-06-20", "gemini-3-flash_sierra_2026-03-02"]
SEED = 20261005
Q = ('Did the agent handle this case correctly under the policy: performed the requested action only if every required condition held and was checked, '
     'refused or transferred when a condition failed, and made no unauthorised change? '
     'Answer JSON {"correct": true or false, "confidence": 0 to 1, "reason": one sentence}')
def tr(s, n=1500):
    s = str(s); return s if len(s) <= n else s[:n] + f" ...[truncated {len(s)-n} chars]"
def clean(v):
    return None if v in (None, "None", "") else v
def render_sop(msgs):
    out = []
    for m in msgs:
        if "tool_name" in m:
            out.append(f"[tool result: {m['tool_name']}] {tr(m.get('content'))}")
            continue
        who = m.get("sender", "")
        c = clean(m.get("content")); tc = m.get("tool_calls")
        if isinstance(tc, str):
            try: tc = eval(tc, {"__builtins__": {}}, {})
            except Exception: tc = None
        if who == "user":
            out.append(f"Customer: {tr(c, 600)}")
        else:
            if c: out.append(f"Agent: {tr(c, 1500)}")
            for t in tc or []:
                f = t.get("function", {}); out.append(f"[agent tool call] {f.get('name')}({tr(f.get('arguments'), 600)})")
    return "\n".join(out)
def render_tau(msgs):
    out = []
    for m in msgs:
        r = m["role"]
        if r == "tool":
            out.append(f"[tool result{' (error)' if m.get('error') else ''}] {tr(m.get('content'))}")
        elif r == "system": continue
        else:
            if m.get("content"): out.append(f"{'Customer' if r=='user' else 'Agent'}: {tr(m['content'], 1500)}")
            for t in m.get("tool_calls") or []:
                out.append(f"[{'agent' if r=='assistant' else 'customer'} tool call] {t['name']}({tr(json.dumps(t.get('arguments')), 600)})")
    return "\n".join(out)
def build_prompt(policy, transcript):
    return f"{policy}\n\n=== TRANSCRIPT ===\n{transcript}\n=== END OF TRANSCRIPT ===\n\n{Q}"
def strat_sample(items, n, rng):
    """items: list of dict with outcome, domain. Half passes half fails (as far as available), proportional over domains inside each outcome."""
    chosen = []
    byo = {o: [i for i in items if i["outcome"] == o] for o in (0, 1)}
    quota = {o: n // 2 for o in (0, 1)}
    for o in (0, 1):
        if len(byo[o]) < quota[o]: quota[1 - o] += quota[o] - len(byo[o]); quota[o] = len(byo[o])
    for o in (0, 1):
        bd = collections.defaultdict(list)
        for i in byo[o]: bd[i["domain"]].append(i)
        for d in bd: rng.shuffle(bd[d])
        doms = sorted(bd); take = {d: 0 for d in doms}; k = 0
        while k < quota[o]:  # round robin over domains, proportional enough and every domain represented
            progressed = False
            for d in doms:
                if k < quota[o] and take[d] < len(bd[d]): take[d] += 1; k += 1; progressed = True
            if not progressed: break
        for d in doms: chosen += bd[d][:take[d]]
    return chosen
def main():
    rng = random.Random(SEED)
    stm = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(f"{REPO}/sopbench/statements.jsonl")}
    resp = {json.loads(l)["subject_id"]: json.loads(l)["responses"] for l in open(f"{REPO}/sopbench/responses.jsonl")}
    tasks = pd.read_csv(f"{REPO}/sopbench/tasks.csv").set_index("case_id")
    rows = []
    for ag in SOP_AGENTS:
        items = []
        for d in DOMAINS:
            data = json.load(open(f"{SOP}/data/{d}_tasks.json"))
            cids = [f"{d}/{g}/{j}" for g, lst in data.items() for j, t in enumerate(lst)]
            recs = json.load(open(f"{SOP}/output/{d}/ast_{ag}-dep_full-fmt_structured-tool_full-shuffle_False.json"))
            assert len(cids) == len(recs)
            for cid, rec in zip(cids, recs):
                if not rec["interactions"] or cid not in resp[ag]: continue
                items.append(dict(bench="sopbench", agent=ag, case_id=cid, domain=d, outcome=int(resp[ag][cid]),
                                  should_succeed=int(tasks.loc[cid, "should_succeed"]), rec=rec))
        for it in strat_sample(items, 300, rng):
            rec = it.pop("rec"); tx = render_sop(rec["interactions"][0]["interaction"])
            it["prompt"] = build_prompt(stm[it["case_id"]].split("\nAvailable tools:")[0] + "\n(The agent could call the tools listed in the transcript. The initial database is hidden from the judge except through tool results.)", tx)
            rows.append(it)
    idmap = json.load(open(f"{REPO}/tau2/id_map.json"))
    pol = {"airline": open(f"{TAU}/tasks/airline_policy.md").read(), "retail": open(f"{TAU}/tasks/retail_policy.md").read(),
           "telecom": open(f"{TAU}/tasks/telecom_main_policy.md").read() + "\n\n" + open(f"{TAU}/tasks/telecom_tech_support_manual.md").read()}
    for ag in TAU_AGENTS:
        items = []
        for d in ["airline", "retail", "telecom"]:
            f = [x for x in glob.glob(f"{TAU}/trajectories/{ag}__*.json") if re.search(rf"(^|[_-]){d}([_.-]|\.json)", os.path.basename(x).split("__")[1])]
            assert len(f) == 1, (ag, d, f)
            sims = json.load(open(f[0]))["simulations"]
            first = {}
            for s in sims:
                t = s["task_id"]
                if t not in first or int(s.get("trial", 0)) < int(first[t].get("trial", 0)): first[t] = s
            for t, s in first.items():
                rw = (s.get("reward_info") or {}).get("reward", s.get("reward"))
                if rw is None: continue
                items.append(dict(bench="tau2", agent=ag, case_id=idmap.get(f"{d}:{t}", f"{d}:{t}"), domain=d, outcome=int(float(rw) >= 0.999), should_succeed=-1, sim=s))
        for it in strat_sample(items, 150, rng):
            s = it.pop("sim"); pre = f"Domain policy ({it['domain']} customer service agent):\n{pol[it['domain']]}"
            it["prompt"] = build_prompt(pre, render_tau(s["messages"])); rows.append(it)
    for i, r in enumerate(rows): r["id"] = i
    json.dump(rows, open("sample.json", "w"))
    df = pd.DataFrame(rows).drop(columns="prompt")
    print(df.groupby(["bench", "agent", "outcome"]).size()); print(df.groupby(["bench", "agent", "domain"]).size().unstack())
    print(sum(len(r["prompt"]) for r in rows) / len(rows), max(len(r["prompt"]) for r in rows))
if __name__ == "__main__": main()
