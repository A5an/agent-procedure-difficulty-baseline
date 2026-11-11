"""Write documentation probes for the 20 test families (PREREG.md). The generator never sees a benchmark task.

Per family: up to 10 generator calls, each with 3 candidate customers; a probe is kept only if every gold action runs on
a fresh environment without a tool error or a failure message. Stops at 6 valid probes. Output probes.json (benchmark
task format) and probes_log.jsonl (every attempt, kept or dropped, with the reason).
"""
import collections, json, random, re, sys, threading, uuid
from concurrent.futures import ThreadPoolExecutor
import requests
from common import HERE, DOM, PROXY, GEN_MODEL, is_write

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import Task
from tau2.domains.indian_banking.environment import get_environment

N_VALID, MAX_ATTEMPTS, SEED = 6, 10, 20261102
MAX_TOK = 8192  # first pass; the top-up pass uses 32768 (57 of 183 first-pass calls hit the 8192 limit while thinking)
sel = json.load(open(HERE / "selection.json"))
DB = json.load(open(DOM / "db.json"))["customers"]
EXCL = set(sel["excluded_customers"])
POLICY = (DOM / "policy.md").read_text()
INSTR = (DOM / "agent_instruction.txt").read_text()
_env0 = get_environment()
TOOLS = [t.openai_schema["function"] for t in _env0.get_tools()]
TOOLS_TXT = json.dumps(TOOLS, indent=0)
lock = threading.Lock()


def need(purpose):
    """products a candidate customer must hold, read from the purpose line only"""
    p = purpose.lower()
    req = {"accounts": 1}
    if "card" in p: req["cards"] = 2 if "both" in p or "two" in p else 1
    if "mandate" in p or "cancel #" in p:
        req["mandates"] = 2 if ("#2" in p or "audit" in p or "all mandates" in p or "match by payee" in p) else 1
    if "insur" in p or "policy check" in p or "policy details" in p: req["insurance"] = 1
    if ("details" in p and ("quote" in p or "renewal" in p)) or "fd closure" in p or "ladder" in p: req["deposits"] = 1
    if "request status" in p or "follow-up" in p or "chase" in p: req["requests"] = 1
    if "offers" in p: req["offers"] = 1
    return req


def candidates(purpose, rng):
    req = need(purpose)
    ok = [cid for cid, c in DB.items() if cid not in EXCL and all(len(c.get(k) or []) >= n for k, n in req.items())]
    return rng.sample(ok, min(3, len(ok))), req


PROMPT = """You write one test conversation for a bank's virtual assistant. A simulated customer will follow your script and
the assistant will be scored by whether it makes exactly the right tool calls and leaves the bank database in the right
state.

PROCEDURE TO TEST (one line describing the steps of this kind of request):
{purpose}

BANK POLICY:
{policy}

ASSISTANT'S OPERATING INSTRUCTIONS:
{instr}

TOOLS THE ASSISTANT HAS (JSON schemas):
{tools}

CANDIDATE CUSTOMERS (full database records; the customer is already logged in):
{customers}

Pick the one candidate whose records make this procedure possible and realistic, and write a script for that customer
that follows the procedure line step by step. Then list the gold actions: the exact tool calls a correct assistant must
make, in order, following the policy (reads included, with the exact IDs from the record). Rules:
- Only IDs and values that appear in the chosen customer's record or that the customer states in the script.
- A write (state-changing call) appears only if the policy allows it after the customer confirms; if the procedure has
  a step the assistant must refuse or that must not be repeated, that call must not appear (again).
- For every write give "compare_args": the argument names that must match (the ones that identify what is done).
- Arguments must fit the tool schemas exactly.
- The customer must not volunteer IDs until the assistant asks; say so in known_info.
- task_instructions: numbered steps for the customer, strictly in order, ending with "End with ###STOP###."
If no candidate fits, answer {{"customer_id": null}}.

Answer with one JSON object only:
{{"customer_id": "...", "persona": "...", "reason_for_call": "...", "known_info": "...", "unknown_info": "...",
 "task_instructions": "...", "gold_actions": [{{"name": "...", "arguments": {{}}, "compare_args": ["..."]}}],
 "nl_assertion": "one sentence describing correct handling"}}"""

def failed(content):
    """the domain signals failure with an error_code object, a non-empty "errors" list, or an "Error: ..." string"""
    try:
        j = json.loads(content)
    except Exception:
        return (content or "").strip().lower().startswith("error")
    if isinstance(j, dict):
        if "error_code" in j: return True
        if isinstance(j.get("errors"), list) and j["errors"]: return True
    return False


def validate(cid, actions):
    env = get_environment()
    env.set_state(Task.model_validate({"id": "v", "user_scenario": {"instructions": "x"},
                                       "initial_state": {"initialization_data": {"agent_data": {"active_customer": cid}}}}
                                      ).initial_state.initialization_data, None, [])
    for a in actions:
        tm = env.get_response(ToolCall(id=str(uuid.uuid4())[:8], name=a["name"], arguments=a.get("arguments") or {},
                                       requestor="assistant"))
        if tm.error: return f"tool error on {a['name']}: {tm.content[:160]}"
        if failed(tm.content): return f"failure message on {a['name']}: {(tm.content or '')[:160]}"
    return None


def gen(fam, have=0, tag=""):
    purpose = sel["purpose"][fam]
    rng = random.Random(f"{SEED}-{fam}{tag}")
    kept, log = [], []
    for att in range(MAX_ATTEMPTS):
        if have + len(kept) >= N_VALID: break
        cands, req = candidates(purpose, rng)
        prompt = PROMPT.format(purpose=purpose, policy=POLICY, instr=INSTR, tools=TOOLS_TXT,
                               customers=json.dumps({c: DB[c] for c in cands}, indent=0))
        rec = {"family": fam, "attempt": att, "candidates": cands, "need": req, "pass": tag or "first"}
        try:
            r = requests.post(f"{PROXY}/chat/completions", timeout=600, json={
                "model": f"gen|{GEN_MODEL}", "temperature": 0.7, "max_tokens": MAX_TOK,
                "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            j = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        except Exception as e:
            rec["drop"] = f"generation failed: {type(e).__name__}: {str(e)[:120]}"; log.append(rec); continue
        cid = j.get("customer_id")
        if not cid or cid not in cands:
            rec["drop"] = "no fitting candidate" if not cid else "customer not among candidates"; log.append(rec); continue
        acts = j.get("gold_actions") or []
        if not acts:
            rec["drop"] = "no gold actions"; log.append(rec); continue
        why = validate(cid, acts)
        if why:
            rec["drop"] = why; rec["gold"] = acts; log.append(rec); continue
        k = have + len(kept)
        actions = []
        for i, a in enumerate(acts):
            x = {"action_id": f"p{fam}{k}_{i}", "name": a["name"], "arguments": a.get("arguments") or {}}
            if is_write(a["name"]) and a.get("compare_args"): x["compare_args"] = list(a["compare_args"])
            actions.append(x)
        task = {"id": f"probe_{fam}_{k:02d}",
                "description": {"purpose": purpose, "notes": f"family={fam}; customer_id={cid}; probe=1"},
                "user_scenario": {"persona": j.get("persona", ""), "instructions": {
                    "domain": "banking", "reason_for_call": j.get("reason_for_call", ""), "known_info": j.get("known_info", ""),
                    "unknown_info": j.get("unknown_info", ""), "task_instructions": j.get("task_instructions", "")}},
                "initial_state": {"initialization_data": {"agent_data": {"active_customer": cid}}},
                "evaluation_criteria": {"actions": actions, "communicate_info": [], "nl_assertions": [j.get("nl_assertion", "")],
                                        "reward_basis": ["ACTION", "DB"]}}
        try:
            Task.model_validate(task)
        except Exception as e:
            rec["drop"] = f"task schema: {str(e)[:120]}"; log.append(rec); continue
        rec["kept"] = task["id"]; log.append(rec); kept.append(task)
    with lock:
        with open(HERE / "probes_log.jsonl", "a") as f:
            for r in log: f.write(json.dumps(r) + "\n")
    print(fam, "kept", len(kept), "of", len(log), "attempts", flush=True)
    return kept


if __name__ == "__main__":
    if sys.argv[1:] == ["topup"]:
        # second pass, before any test episode: families short of 6 probes because the first pass ran out of tokens
        MAX_TOK = 32768
        probes = json.load(open(HERE / "probes.json"))
        have = collections.Counter(p["id"].split("_", 1)[1].rsplit("_", 1)[0] for p in probes)
        short = [f for f in sel["test_families"] if have[f] < N_VALID]
        with ThreadPoolExecutor(6) as ex:
            res = list(ex.map(lambda f: gen(f, have[f], "-topup"), short))
        probes += [t for r in res for t in r]
        json.dump(probes, open(HERE / "probes.json", "w"), indent=1)
        print("probes", len(probes))
        sys.exit()
    fams = sys.argv[1:] or sel["test_families"]
    with ThreadPoolExecutor(5) as ex:
        res = list(ex.map(gen, fams))
    probes = [t for r in res for t in r]
    path = HERE / ("probes.json" if not sys.argv[1:] else "probes_test.json")
    json.dump(probes, open(path, "w"), indent=1)
    print("probes", len(probes), "->", path)
