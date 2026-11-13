"""E1 probe generation (PREREG.md). Conditions: k1 (first of the two existing seeds), k5 (the two seeds plus three more
tasks of the family outside its evaluation tasks), foreign (the two seeds of another family, rule in select_seedk.py).
Generator, prompt, candidate filter, validity check and compare_args convention are those of ../tib/seeded/gen_seeded.py;
only the examples block changes (its count words, and for foreign the header says the examples are of another
procedure). Calls go through this folder's proxy (port 8095, ledger and cap here).

  <tau venv python> gen_seedk.py <k1|k5|foreign> [family ...]   (default: the pilot families)
Per family: probes/<cond>/<family>.json (kept probes, benchmark task format) and probes_log.jsonl (every attempt)."""
import json, random, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tib")); sys.path.insert(0, str(HERE.parent / "tib" / "seeded"))
import gen_probes as G          # prompt, policy, instructions, tool schemas, customer db, need(), validate()
import gen_seeded as S          # ALL tasks, apply_convention()
from common import is_write
from tau2.data_model.tasks import Task

PROXY = "http://127.0.0.1:8095/v1"
SEL = json.load(open(HERE / "selection_seedk.json"))
FAM = SEL["families"]
EXCL = {k: set(v) for k, v in SEL["excluded_customers"].items()}
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
lock = threading.Lock()

SAME = """

PAST CONVERSATIONS OF THIS PROCEDURE ({n} {scripts} that real customers followed, and the tool calls a correct assistant
made for {them}; {they} belong to other customers, so their IDs do not apply to yours):
{examples}

Write a NEW conversation for the customer you pick, in the same style as {these}: the same kind of request, the
same way the customer behaves (what they do and do not volunteer, the order they insist on, any trap or correction in
the request), adapted to your customer's records. Do not copy IDs or amounts from the examples."""

FOREIGN = """

PAST CONVERSATIONS OF ANOTHER PROCEDURE OF THIS BANK ({n} {scripts} that real customers followed for a different
request, and the tool calls a correct assistant made for {them}; {they} belong to other customers, so their IDs do not
apply to yours):
{examples}

Write a NEW conversation for the customer you pick, for the procedure described above (not the procedure of these
examples), in the same style as {these}: the same way customers of this bank behave (what they do and do not volunteer,
the order they insist on, any correction in the request), adapted to your procedure and your customer's records. Do not
copy IDs or amounts from the examples."""


def block(cond, fam):
    ids = FAM[fam]["seeds_" + cond]
    ex = [{"customer_script": S.ALL[t]["user_scenario"]["instructions"],
           "correct_tool_calls": [{"name": a["name"], "arguments": a["arguments"]} for a in S.ALL[t]["evaluation_criteria"]["actions"]]}
          for t in ids]
    n = len(ids)
    w = dict(n=WORD[n], scripts="script" if n == 1 else "scripts", them="it" if n == 1 else "them",
             they="it" if n == 1 else "they", these="this past one" if n == 1 else "these past ones", examples=json.dumps(ex, indent=1))
    if n == 1:   # "real customers followed" -> singular
        return SAME.format(**w).replace("that real customers followed", "that a real customer followed").replace(
            "they belong to other customers", "it belongs to another customer").replace("it belong to other customers, so their", "it belongs to another customer, so its")
    return (FOREIGN if cond == "foreign" else SAME).format(**w)


def candidates(fam, rng):
    req = G.need(FAM[fam]["purpose"]); ex = EXCL[FAM[fam]["source"]]
    ok = [cid for cid, c in G.DB.items() if cid not in ex and all(len(c.get(k) or []) >= n for k, n in req.items())]
    return rng.sample(ok, min(3, len(ok)))


def gen(cond, fam):
    outp = HERE / "probes" / cond / f"{fam}.json"
    if outp.exists():
        print(cond, fam, "already done", flush=True); return
    purpose = FAM[fam]["purpose"]
    rng = random.Random(f"20261112-{cond}-{fam}")
    kept, log = [], []
    for att in range(G.MAX_ATTEMPTS):
        if len(kept) >= G.N_VALID: break
        cands = candidates(fam, rng)
        prompt = G.PROMPT.format(purpose=purpose, policy=G.POLICY, instr=G.INSTR, tools=G.TOOLS_TXT,
                                 customers=json.dumps({c: G.DB[c] for c in cands}, indent=0)) + block(cond, fam)
        rec = {"family": fam, "cond": cond, "attempt": att, "candidates": cands}
        try:
            r = requests.post(f"{PROXY}/chat/completions", timeout=900, json={
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
        actions = [{"action_id": f"{cond}{fam}{k}_{i}", "name": a["name"], "arguments": a.get("arguments") or {},
                    **({"compare_args": list(a["compare_args"])} if is_write(a["name"]) and a.get("compare_args") else {})}
                   for i, a in enumerate(acts)]
        task = {"id": f"{cond}_{fam}_{k:02d}",
                "description": {"purpose": purpose, "notes": f"family={fam}; customer_id={cid}; cond={cond}"},
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
        for x in log: f.write(json.dumps(x) + "\n")
    stopped = any("cap" in (x.get("drop") or "") or "429" in (x.get("drop") or "") for x in log)
    if not stopped:
        outp.parent.mkdir(parents=True, exist_ok=True); json.dump(kept, open(outp, "w"), indent=1)
    print(cond, fam, "kept", len(kept), "of", len(log), "(stopped by cap/429, not saved)" if stopped else "", flush=True)


if __name__ == "__main__":
    cond = sys.argv[1]; assert cond in ("k1", "k5", "foreign")
    fams = sys.argv[2:] or SEL["pilot_families"]
    with ThreadPoolExecutor(5) as ex:
        list(ex.map(lambda f: gen(cond, f), fams))
