"""Probes for the 30 replication families (PREREG.md): seeded (as Exp 3) and documentation-only (as P0-2), same
prompts, same candidate filter and gold-action check, through the proxy of this folder.

  python gen_rep.py seeded|doc  -> probes_seeded.json / probes_doc.json, probes_log.jsonl
Differences from the originals, fixed here: max_tokens 32768 from the first call (P0-2 needed a top-up pass at 32768
after truncations at 8192); documentation probes get the benchmark's per-tool compare_args convention at generation
(P0-2 applied it afterwards as scoring v2; Exp 3 already did it at generation).
"""
import json, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
T = HERE.parent / "tib"
sys.path.insert(0, str(T)); sys.path.insert(0, str(T / "seeded"))
import gen_probes as G      # prompt, policy, tool schemas, candidate filter, gold-action check
import gen_seeded as S      # the seeded-probe instruction block and the compare_args convention
from common import is_write
from tau2.data_model.tasks import Task

sel = json.load(open(HERE / "selection_rep.json"))
G.EXCL = set(sel["excluded_customers"])
lock = threading.Lock()


def examples(fam):
    out = []
    for tid in sel["seed_tasks"][fam]:
        t = S.ALL[tid]
        out.append({"customer_script": t["user_scenario"]["instructions"],
                    "correct_tool_calls": [{"name": a["name"], "arguments": a["arguments"]} for a in t["evaluation_criteria"]["actions"]]})
    return json.dumps(out, indent=1)


def gen(fam, kind):
    purpose = sel["purpose"][fam]
    rng = random.Random(f"20261104-rep-{fam}-{kind}")
    kept, log = [], []
    for att in range(G.MAX_ATTEMPTS):
        if len(kept) >= G.N_VALID:
            break
        cands, req = G.candidates(purpose, rng)
        prompt = G.PROMPT.format(purpose=purpose, policy=G.POLICY, instr=G.INSTR, tools=G.TOOLS_TXT,
                                 customers=json.dumps({c: G.DB[c] for c in cands}, indent=0))
        if kind == "seeded":
            prompt += S.EXTRA.format(examples=examples(fam))
        rec = {"family": fam, "kind": kind, "attempt": att, "candidates": cands}
        try:
            r = requests.post(f"{G.PROXY}/chat/completions", timeout=900, json={
                "model": f"gen|{G.GEN_MODEL}", "temperature": 0.7, "max_tokens": 32768,
                "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            j = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        except Exception as e:
            rec["drop"] = f"generation failed: {type(e).__name__}: {str(e)[:120]}"; log.append(rec)
            if "cap" in str(e) or "429" in str(e):
                break
            continue
        cid = j.get("customer_id")
        if not cid or cid not in cands:
            rec["drop"] = "no fitting candidate" if not cid else "customer not among candidates"; log.append(rec); continue
        acts = j.get("gold_actions") or []
        why = G.validate(cid, acts) if acts else "no gold actions"
        if why:
            rec["drop"] = why; log.append(rec); continue
        k = len(kept)
        pre = "s" if kind == "seeded" else "p"
        actions = [{"action_id": f"{pre}{fam}{k}_{i}", "name": a["name"], "arguments": a.get("arguments") or {},
                    **({"compare_args": list(a["compare_args"])} if is_write(a["name"]) and a.get("compare_args") else {})}
                   for i, a in enumerate(acts)]
        task = {"id": f"{'seeded' if kind == 'seeded' else 'probe'}_{fam}_{k:02d}",
                "description": {"purpose": purpose, "notes": f"family={fam}; customer_id={cid}; {kind}=1"},
                "user_scenario": {"persona": j.get("persona", ""), "instructions": {
                    "domain": "banking", "reason_for_call": j.get("reason_for_call", ""), "known_info": j.get("known_info", ""),
                    "unknown_info": j.get("unknown_info", ""), "task_instructions": j.get("task_instructions", "")}},
                "initial_state": {"initialization_data": {"agent_data": {"active_customer": cid}}},
                "evaluation_criteria": {"actions": S.apply_convention(actions), "communicate_info": [],
                                        "nl_assertions": [j.get("nl_assertion", "")], "reward_basis": ["ACTION", "DB"]}}
        try:
            Task.model_validate(task)
        except Exception as e:
            rec["drop"] = f"task schema: {str(e)[:120]}"; log.append(rec); continue
        rec["kept"] = task["id"]; log.append(rec); kept.append(task)
    with lock, open(HERE / "probes_log.jsonl", "a") as f:
        for x in log:
            f.write(json.dumps(x) + "\n")
    print(kind, fam, "kept", len(kept), "of", len(log), flush=True)
    return kept


if __name__ == "__main__":
    kind = sys.argv[1]
    fams = sys.argv[2:] or sel["test_families"]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda f: gen(f, kind), fams))
    probes = [t for r in res for t in r]
    out = HERE / (f"probes_{kind}.json" if not sys.argv[2:] else f"probes_{kind}_test.json")
    json.dump(probes, open(out, "w"), indent=1)
    print("probes", len(probes), "->", out)
