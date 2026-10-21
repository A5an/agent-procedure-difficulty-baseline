"""All gpt-4.1-mini cases: 830 SOPBench, first trial of every tau2 case. Same prompt builders as prep.py."""
import json, re, glob, os
import pandas as pd
import prep as p
def main():
    stm = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(f"{p.REPO}/sopbench/statements.jsonl")}
    ag = "gpt-4.1-mini-mode_fc"
    resp = {json.loads(l)["subject_id"]: json.loads(l)["responses"] for l in open(f"{p.REPO}/sopbench/responses.jsonl")}[ag]
    tasks = pd.read_csv(f"{p.REPO}/sopbench/tasks.csv").set_index("case_id"); rows = []
    for d in p.DOMAINS:
        data = json.load(open(f"{p.SOP}/data/{d}_tasks.json")); cids = [f"{d}/{g}/{j}" for g, lst in data.items() for j, t in enumerate(lst)]
        recs = json.load(open(f"{p.SOP}/output/{d}/ast_{ag}-dep_full-fmt_structured-tool_full-shuffle_False.json"))
        for cid, rec in zip(cids, recs):
            if not rec["interactions"] or cid not in resp: continue
            tx = p.render_sop(rec["interactions"][0]["interaction"])
            rows.append(dict(bench="sopbench", agent=ag, case_id=cid, domain=d, outcome=int(resp[cid]), should_succeed=int(tasks.loc[cid, "should_succeed"]),
                prompt=p.build_prompt(stm[cid].split("\nAvailable tools:")[0] + "\n(The agent could call the tools listed in the transcript. The initial database is hidden from the judge except through tool results.)", tx)))
    idmap = json.load(open(f"{p.REPO}/tau2/id_map.json"))
    pol = {"airline": open(f"{p.TAU}/tasks/airline_policy.md").read(), "retail": open(f"{p.TAU}/tasks/retail_policy.md").read(),
           "telecom": open(f"{p.TAU}/tasks/telecom_main_policy.md").read() + "\n\n" + open(f"{p.TAU}/tasks/telecom_tech_support_manual.md").read()}
    tag = p.TAU_AGENTS[0]
    for d in ["airline", "retail", "telecom"]:
        f = [x for x in glob.glob(f"{p.TAU}/trajectories/{tag}__*.json") if re.search(rf"(^|[_-]){d}([_.-]|\.json)", os.path.basename(x).split("__")[1])]; assert len(f) == 1
        first = {}
        for s in json.load(open(f[0]))["simulations"]:
            t = s["task_id"]
            if t not in first or int(s.get("trial", 0)) < int(first[t].get("trial", 0)): first[t] = s
        for t, s in first.items():
            rw = (s.get("reward_info") or {}).get("reward", s.get("reward"))
            if rw is None: continue
            rows.append(dict(bench="tau2", agent=tag, case_id=idmap.get(f"{d}:{t}", f"{d}:{t}"), domain=d, outcome=int(float(rw) >= 0.999), should_succeed=-1,
                prompt=p.build_prompt(f"Domain policy ({d} customer service agent):\n{pol[d]}", p.render_tau(s["messages"]))))
    for i, r in enumerate(rows): r["id"] = i
    json.dump(rows, open("full.json", "w")); print(pd.DataFrame(rows).groupby(["bench", "outcome"]).size())
main()
