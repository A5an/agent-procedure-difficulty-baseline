"""Build tau2-bench (airline, retail, telecom) and tau-Knowledge banking inputs for the baseline.

Inputs (fetched by src/download_data.py into external_data/): task files of
github.com/sierra-research/tau2-bench and the leaderboard trajectories of its public bucket
sierra-tau-bench-public. Writes data/tau2/ and data/tauk_banking/ in the same layout as data/sopbench/, except that the
response matrix keeps every attempt: responses.jsonl cells are {"successes": k, "trials": n},
the binomial format agent-psychometrics supports (their Terminal-Bench per-attempt path).

Outcome: reward >= 0.999 counts as success (rewards in these files are 0 or 1).
Agents: one subject per leaderboard submission (file prefix before "__").

Task statement (leak-free as far as the task record allows): the customer scenario written for the
user simulator, i.e. reason for call, known info, unknown info and task instructions (tau2), or the
instruction text (banking). The agent never sees this text verbatim; it is the task description,
as the issue text is for SWE-bench. Never used: description.purpose (states what the test checks),
evaluation_criteria (gold actions and assertions), initial_state (hidden environment state),
annotations, required_documents (banking reference annotation), persona, and the telecom task ids,
which list the injected device faults. Task ids are replaced by opaque ids; the map is kept.
"""

import collections
import csv
import json
import re

from common import DATA, EXTERNAL as EXT


def load_results(path):
    d = json.loads(path.read_text())
    by = collections.defaultdict(dict)
    for s in d.get("simulations") or d.get("results") or []:
        tid = s.get("task_id")
        trial = s.get("trial", s.get("trial_id", 0))
        rw = (s.get("reward_info") or {}).get("reward", s.get("reward"))
        if tid is None or rw is None:
            continue
        by[str(tid)][int(trial)] = float(rw) >= 0.999
    return by


def scenario_text(domain, task):
    ins = task["user_scenario"]["instructions"]
    if isinstance(ins, str):
        body = ins.strip()
    else:
        parts = [("Reason for call", ins.get("reason_for_call")), ("Known info", ins.get("known_info")),
                 ("Unknown info", ins.get("unknown_info")), ("Instructions", ins.get("task_instructions"))]
        body = "\n".join(f"{k}: {v.strip()}" for k, v in parts if v)
    return f"Domain: {domain.replace('_', ' ')} customer service.\nCustomer scenario:\n{body}"


def write(out, rows, results, name):
    """rows: list of (orig_id, domain, group, text); results: {subject: {orig_id: {trial: bool}}}."""
    out.mkdir(parents=True, exist_ok=True)
    opaque = {r[0]: f"{r[1]}/t{i:04d}" for i, r in enumerate(rows)}
    with open(out / "responses.jsonl", "w") as f:
        for subj in sorted(results):
            cells = {}
            for orig, trials in results[subj].items():
                if orig in opaque and trials:
                    cells[opaque[orig]] = {"successes": int(sum(trials.values())), "trials": len(trials)}
            f.write(json.dumps({"subject_id": subj, "responses": cells}) + "\n")
    with open(out / "tasks.csv", "w", newline="") as f, open(out / "statements.jsonl", "w") as g:
        w = csv.writer(f)
        w.writerow(["case_id", "domain", "procedure", "procedure_id", "statement_chars"])
        for orig, dom, group, text in rows:
            w.writerow([opaque[orig], dom, group, f"{dom}/{group}", len(text)])
            g.write(json.dumps({"case_id": opaque[orig], "text": text}) + "\n")
    (out / "id_map.json").write_text(json.dumps(opaque, indent=1))
    n_cells = sum(len(v) for v in results.values())
    trials = [len(t) for v in results.values() for t in v.values()]
    print(name, "tasks", len(rows), "subjects", len(results), "cells", n_cells,
          "attempts", sum(trials), "cells with <4 attempts", sum(t < 4 for t in trials))


def tau2():
    rows, results = [], collections.defaultdict(dict)
    for dom in ["airline", "retail", "telecom"]:
        repo = json.loads((EXT / "tau2-domains/tasks" / f"{dom}_tasks.json").read_text())
        repo = repo if isinstance(repo, list) else repo["tasks"]
        tasks = {str(t["id"]): t for t in repo}
        run_ids = set()
        for f in sorted((EXT / "tau2-domains/trajectories").glob("*.json")):
            if not re.search(rf"(^|[_-]){dom}([_.-]|$)", f.name):
                continue
            by = load_results(f)
            subj = f.name.split("__")[0]
            for tid, tr in by.items():
                results[subj][f"{dom}:{tid}"] = tr
            run_ids |= set(by)
        for tid in sorted(run_ids, key=lambda x: (len(x), x)):
            t = tasks[tid]
            # telecom tasks come in 3 resolution paths; airline and retail tasks are all distinct
            purpose = (t.get("description") or {}).get("purpose") or ""
            group = re.sub(r"\W+", "_", purpose).strip("_") if dom == "telecom" else f"task_{tid}"
            rows.append((f"{dom}:{tid}", dom, group, scenario_text(dom, t)))
    write(DATA / "tau2", rows, results, "tau2")


def banking():
    rows, results = [], collections.defaultdict(dict)
    for p in sorted((EXT / "tau2-banking/tasks").glob("task_*.json")):
        t = json.loads(p.read_text())
        rows.append((str(t["id"]), "banking_knowledge", f"task_{t['id']}", scenario_text("banking_knowledge", t)))
    for f in sorted((EXT / "tau2-banking/trajectories").glob("*.json")):
        for tid, tr in load_results(f).items():
            results[f.name.split("__")[0]][tid] = tr
    write(DATA / "tauk_banking", rows, results, "tauk_banking")


if __name__ == "__main__":
    tau2()
    banking()
