"""Exp 8 probes (PREREG.md): Claude Opus writes DOC, SEED and PSEUDO probes for the 20 P0-2 families.

  python gen_claude.py pseudo_convs | doc | seed | pseudo
pseudo_convs: per family 2 synthetic past conversations (script + correct tool calls, executed for validity), no
benchmark task seen. The probe kinds reuse the P0-2 prompt (../tib/gen_probes.py) and the Exp 3 seeded block.
"""
import json, random, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
T = HERE.parent / "tib"
sys.path.insert(0, str(T)); sys.path.insert(0, str(T / "seeded"))
import gen_probes as G
import gen_seeded as S
from common import is_write
from tau2.data_model.tasks import Task

SEL = G.sel
FAMS = SEL["test_families"]
lock = threading.Lock()
MODEL = "opus"


def claude(prompt):
    cmd = ["claude", "-p", "--model", MODEL, "--tools", "", "--setting-sources", "", "--no-session-persistence",
           "--output-format", "json", "--system-prompt", "You write test material for a bank's virtual assistant. Output one JSON object only."]
    for a in range(3):
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=1200, cwd="/tmp")
        try:
            d = json.loads(r.stdout)
            if d.get("result") and not d.get("is_error"):
                t = d["result"]; return json.loads(t[t.find("{"): t.rfind("}") + 1])
        except Exception:
            pass
        time.sleep(10 * (a + 1))
    raise RuntimeError("claude failed")


PSEUDO_ASK = """

Instead of a test, write ONE realistic PAST conversation of this procedure, as it happened with a real customer of this
bank: how the customer actually phrased the request, what they did and did not say up front, what they asked in which
order, any correction or trap, and the exact tool calls a correct assistant made. Use the same JSON format as above
(the script fields describe what this customer did). Make it feel like a real support ticket, not a template."""


def gen_pseudo(fam):
    rng = random.Random(f"exp8-{fam}-pseudo-convs")
    convs, tries = [], 0
    while len(convs) < 2 and tries < 8:
        tries += 1
        cands, _ = G.candidates(SEL["purpose"][fam], rng)
        p = G.PROMPT.format(purpose=SEL["purpose"][fam], policy=G.POLICY, instr=G.INSTR, tools=G.TOOLS_TXT,
                            customers=json.dumps({c: G.DB[c] for c in cands}, indent=0)) + PSEUDO_ASK
        try:
            j = claude(p)
        except Exception:
            continue
        cid, acts = j.get("customer_id"), j.get("gold_actions") or []
        if cid in cands and acts and not G.validate(cid, acts):
            convs.append({"customer_script": {k: j.get(k, "") for k in ("reason_for_call", "known_info", "unknown_info", "task_instructions")},
                          "correct_tool_calls": [{"name": a["name"], "arguments": a.get("arguments") or {}} for a in acts]})
    print("pseudo convs", fam, len(convs), "in", tries, flush=True)
    return fam, convs


def gen(fam, kind, pseudo):
    purpose = SEL["purpose"][fam]
    rng = random.Random(f"exp8-{fam}-{kind}")
    kept, log = [], []
    for att in range(10):
        if len(kept) >= 6: break
        cands, _ = G.candidates(purpose, rng)
        prompt = G.PROMPT.format(purpose=purpose, policy=G.POLICY, instr=G.INSTR, tools=G.TOOLS_TXT,
                                 customers=json.dumps({c: G.DB[c] for c in cands}, indent=0))
        if kind == "seed":
            prompt += S.EXTRA.format(examples=S.examples(fam))
        elif kind == "pseudo":
            prompt += S.EXTRA.format(examples=json.dumps(pseudo[fam], indent=1))
        rec = {"family": fam, "kind": kind, "attempt": att}
        try:
            j = claude(prompt)
        except Exception as e:
            rec["drop"] = f"generation failed {e}"; log.append(rec); continue
        cid, acts = j.get("customer_id"), j.get("gold_actions") or []
        if not cid or cid not in cands:
            rec["drop"] = "no candidate"; log.append(rec); continue
        why = G.validate(cid, acts) if acts else "no gold actions"
        if why:
            rec["drop"] = why; log.append(rec); continue
        k = len(kept)
        actions = [{"action_id": f"{kind}{fam}{k}_{i}", "name": a["name"], "arguments": a.get("arguments") or {},
                    **({"compare_args": list(a["compare_args"])} if is_write(a["name"]) and a.get("compare_args") else {})}
                   for i, a in enumerate(acts)]
        task = {"id": f"{kind}_{fam}_{k:02d}", "description": {"purpose": purpose, "notes": f"family={fam}; customer_id={cid}; exp8 {kind}"},
                "user_scenario": {"persona": j.get("persona", ""), "instructions": {"domain": "banking", **{x: j.get(x, "") for x in ("reason_for_call", "known_info", "unknown_info", "task_instructions")}}},
                "initial_state": {"initialization_data": {"agent_data": {"active_customer": cid}}},
                "evaluation_criteria": {"actions": S.apply_convention(actions), "communicate_info": [],
                                        "nl_assertions": [j.get("nl_assertion", "")], "reward_basis": ["ACTION", "DB"]}}
        try:
            Task.model_validate(task)
        except Exception as e:
            rec["drop"] = f"schema {str(e)[:80]}"; log.append(rec); continue
        rec["kept"] = task["id"]; log.append(rec); kept.append(task)
    with lock, open(HERE / "probes_log.jsonl", "a") as f:
        for r in log: f.write(json.dumps(r) + "\n")
    print(kind, fam, "kept", len(kept), "of", len(log), flush=True)
    return kept


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "pseudo_convs":
        with ThreadPoolExecutor(4) as ex:
            res = dict(ex.map(gen_pseudo, FAMS))
        json.dump(res, open(HERE / "pseudo_convs.json", "w"), indent=1); sys.exit()
    pseudo = json.load(open(HERE / "pseudo_convs.json")) if kind == "pseudo" else None
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda f: gen(f, kind, pseudo), FAMS))
    json.dump([t for r in res for t in r], open(HERE / f"probes_{kind}.json", "w"), indent=1)
