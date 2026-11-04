"""Exp 3: probes written with two past conversations of the family as examples (PREREG.md in this folder)."""
import json, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import gen_probes as G                      # reuses the candidate filter, validity check, tool schemas
from common import load_tasks, is_write
from tau2.data_model.tasks import Task

seeds = json.load(open(HERE / "seeds.json"))["seed_tasks"]
conv = json.load(open(HERE.parent / "compare_convention.json"))
ALL = {t["id"]: t for t in load_tasks()}
lock = threading.Lock()

EXTRA = """

PAST CONVERSATIONS OF THIS PROCEDURE (two scripts that real customers followed, and the tool calls a correct assistant
made for them; they belong to other customers, so their IDs do not apply to yours):
{examples}

Write a NEW conversation for the customer you pick, in the same style as these past ones: the same kind of request, the
same way the customer behaves (what they do and do not volunteer, the order they insist on, any trap or correction in
the request), adapted to your customer's records. Do not copy IDs or amounts from the examples."""


def examples(fam):
    out = []
    for tid in seeds[fam]:
        t = ALL[tid]
        out.append({"customer_script": t["user_scenario"]["instructions"],
                    "correct_tool_calls": [{"name": a["name"], "arguments": a["arguments"]} for a in t["evaluation_criteria"]["actions"]]})
    return json.dumps(out, indent=1)


def apply_convention(actions):
    for a in actions:
        rule = conv.get(a["name"])
        if rule is None: continue
        if rule == "__all__": a.pop("compare_args", None)
        else: a["compare_args"] = json.loads(rule)
    return actions


def gen(fam):
    purpose = G.sel["purpose"][fam]
    rng = random.Random(f"{G.SEED}-{fam}-seeded")
    kept, log = [], []
    for att in range(G.MAX_ATTEMPTS):
        if len(kept) >= G.N_VALID: break
        cands, req = G.candidates(purpose, rng)
        prompt = G.PROMPT.format(purpose=purpose, policy=G.POLICY, instr=G.INSTR, tools=G.TOOLS_TXT,
                                 customers=json.dumps({c: G.DB[c] for c in cands}, indent=0)) + EXTRA.format(examples=examples(fam))
        rec = {"family": fam, "attempt": att, "candidates": cands}
        try:
            r = requests.post(f"{G.PROXY}/chat/completions", timeout=900, json={
                "model": f"gen|{G.GEN_MODEL}", "temperature": 0.7, "max_tokens": 32768,
                "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            j = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        except Exception as e:
            rec["drop"] = f"generation failed: {type(e).__name__}: {str(e)[:120]}"; log.append(rec)
            if "cap" in str(e) or "429" in str(e): break
            continue
        cid = j.get("customer_id")
        if not cid or cid not in cands:
            rec["drop"] = "no fitting candidate" if not cid else "customer not among candidates"; log.append(rec); continue
        acts = j.get("gold_actions") or []
        why = G.validate(cid, acts) if acts else "no gold actions"
        if why:
            rec["drop"] = why; log.append(rec); continue
        k = len(kept)
        actions = [{"action_id": f"s{fam}{k}_{i}", "name": a["name"], "arguments": a.get("arguments") or {},
                    **({"compare_args": list(a["compare_args"])} if is_write(a["name"]) and a.get("compare_args") else {})}
                   for i, a in enumerate(acts)]
        task = {"id": f"seeded_{fam}_{k:02d}",
                "description": {"purpose": purpose, "notes": f"family={fam}; customer_id={cid}; seeded=1"},
                "user_scenario": {"persona": j.get("persona", ""), "instructions": {
                    "domain": "banking", "reason_for_call": j.get("reason_for_call", ""), "known_info": j.get("known_info", ""),
                    "unknown_info": j.get("unknown_info", ""), "task_instructions": j.get("task_instructions", "")}},
                "initial_state": {"initialization_data": {"agent_data": {"active_customer": cid}}},
                "evaluation_criteria": {"actions": apply_convention(actions), "communicate_info": [],
                                        "nl_assertions": [j.get("nl_assertion", "")], "reward_basis": ["ACTION", "DB"]}}
        try:
            Task.model_validate(task)
        except Exception as e:
            rec["drop"] = f"task schema: {str(e)[:120]}"; log.append(rec); continue
        rec["kept"] = task["id"]; log.append(rec); kept.append(task)
    with lock, open(HERE / "probes_log.jsonl", "a") as f:
        for r in log: f.write(json.dumps(r) + "\n")
    print(fam, "kept", len(kept), "of", len(log), flush=True)
    return kept


if __name__ == "__main__":
    fams = sys.argv[1:] or G.sel["test_families"]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(gen, fams))
    probes = [t for r in res for t in r]
    out = HERE / ("probes.json" if not sys.argv[1:] else "probes_test.json")
    json.dump(probes, open(out, "w"), indent=1)
    print("probes", len(probes), "->", out)
