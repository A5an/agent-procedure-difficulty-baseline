"""Generate synthetic cases for every policy variant of the 68 usable SOPBench procedures (one LLM call per variant, 10 scenarios),
validate with code (env's own checks), keep up to 6 valid per variant (class-balanced round-robin), write SOPBench task files."""
import sys, os, json, copy, inspect, re, collections, hashlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gt, gen, validate
H = os.path.dirname(os.path.abspath(__file__))
EXCL = {("bank", "cancel_credit_card"), ("bank", "pay_bill_with_credit_card")}
DOMS = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]
FMT_NOTE = ("\nFormat rule: write every date and time in exactly the format of the format examples in the schema. A field whose example is a plain "
            "date such as \"2023-10-01\" must be a plain date with no time of day (no \"T10:00:00\").\n")

def variants():
    out = []
    for dom in DOMS:
        d = json.load(open(f"{gt.SOP}/data/{dom}_tasks.json"))
        for proc, cs in d.items():
            if (dom, proc) in EXCL: continue
            g = collections.OrderedDict()
            for c in cs: g.setdefault(json.dumps(c["constraints_original"], sort_keys=True), []).append(c)
            for vi, (k, L) in enumerate(g.items()): out.append((dom, proc, vi, L[0], len(L)))
    return out

def build_prompt(dom, proc, ref):
    instr, tools = gt.policy_prompt(dom, ref, goal=proc)
    schema = gen.domain_schema(dom, True)
    desc = gen.descriptions(dom)
    ukeys = sorted(ref["user_known"].keys())
    p = f"""You are building test scenarios for a customer-service agent that follows a written policy and has tools that read and change a database.

The agent sees exactly this policy text:
<<<POLICY
{instr}
POLICY>>>

The database schema (field names and value types only, "<key>" stands for an arbitrary identifier such as a username or an id, several entries are allowed under it):
{json.dumps(schema, indent=1)}

Field descriptions: {json.dumps(desc)}

Task. Write 10 different realistic scenarios in which a customer asks the agent to do this one action: `{proc}`.
Each scenario has:
- "initial_database": the full database before the conversation, following the schema exactly (same field names, nesting and value types). Keep it small, 1 to 3 entries per collection, but complete.
- "user_known": the facts the customer knows and gives to the agent. The keys must be exactly these: {ukeys}. Use the credential value the customer would present, it may be right or wrong. Put the values the action needs.
- "user_prompt": the first message of the customer, 1 to 4 sentences, in natural language, which states the request and mentions every value in "user_known".
- "should_perform": true if, given the policy and the database, the agent should carry out `{proc}`, false if the agent must refuse because some condition of the policy is not met. Think about the exact policy conditions that apply to `{proc}`.
Use only values and rules that follow from the policy and the schema. Customers ask for the action in every case, whether or not it is allowed.
Make the scenarios different from each other. Across the 10 scenarios let each condition of the policy for `{proc}` be the one that fails at least once, and let some scenarios have every condition satisfied.
""" + FMT_NOTE + 'Return only a JSON list of 10 objects with the keys "initial_database", "user_known", "user_prompt", "should_perform".'
    return p

_datefmt = {}
def plain_date_keys(dom):
    """keys whose value is a plain YYYY-MM-DD in every real database of the domain"""
    if dom in _datefmt: return _datefmt[dom]
    seen = collections.defaultdict(set)
    def walk(v):
        if isinstance(v, dict):
            for k, x in v.items():
                if isinstance(x, str) and re.match(r"^\d{4}-\d{2}-\d{2}", x): seen[k].add(bool(re.match(r"^\d{4}-\d{2}-\d{2}$", x)))
                walk(x)
        elif isinstance(v, list):
            for x in v: walk(x)
    for cs in json.load(open(f"{gt.SOP}/data/{dom}_tasks.json")).values():
        for c in cs: walk(c["initial_database"])
    _datefmt[dom] = {k for k, s in seen.items() if s == {True}}
    return _datefmt[dom]

def repair(dom, v, stats):
    keys = plain_date_keys(dom)
    if isinstance(v, dict):
        for k, x in list(v.items()):
            if k in keys and isinstance(x, str) and re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d", x): v[k] = x[:10]; stats["date_repaired"] += 1
            else: repair(dom, x, stats)
    elif isinstance(v, list):
        for x in v: repair(dom, x, stats)

def job(a):
    dom, proc, vi, ref, nreal = a
    prompt = build_prompt(dom, proc, ref)
    rec = gen.call(prompt, 0.7, 0)
    return {"dom": dom, "proc": proc, "vi": vi, "nreal": nreal, "key": rec["key"], "finish": rec["finish"], "usage": rec["usage"], "text": rec["text"]}

def validate_case(dom, proc, ref, c, stats):
    r = {}
    try:
        db = c["initial_database"]; uk = c["user_known"]; msg = c["user_prompt"]
        assert isinstance(db, dict) and isinstance(uk, dict) and isinstance(msg, str)
    except Exception:
        r["ok"] = False; r["why"] = "keys"; return r
    repair(dom, db, stats)
    if set(uk.keys()) != set(ref["user_known"].keys()): r["ok"] = False; r["why"] = "uk_keys"; return r
    if set(db.keys()) != set(ref["initial_database"].keys()): r["ok"] = False; r["why"] = "top_keys"; return r
    case = copy.deepcopy(ref); case["initial_database"] = db; case["user_known"] = uk; case["user_prompt"] = msg
    ok, err = gt.load_check(dom, case, goal=proc)
    if not ok: r["ok"] = False; r["why"] = "load"; return r
    try:
        dep_innate, default_dep, _ = gt.defaults(dom)
        dep_full = copy.deepcopy(default_dep); dep_full[proc] = case["constraints"]
        st = gt.domain_keys[dom + "_strict"](copy.deepcopy(db), dep_innate, dep_full, case["constraint_parameters"])
        params = [p for p, v in inspect.signature(getattr(st.evaluation_get_domain_system(), proc)).parameters.items()
                  if v.default is inspect.Parameter.empty and v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD)]
        if not all(p in uk for p in params): r["ok"] = False; r["why"] = "params"; return r
        if "username" in uk and hasattr(st, "login_user"):
            lp = [p for p in inspect.signature(st.login_user).parameters]
            if all(p in uk for p in lp): st.login_user(**{p: uk[p] for p in lp})
        getattr(st, proc)(**{p: uk[p] for p in params})
    except Exception as e:
        r["ok"] = False; r["why"] = "tool_err"; return r
    try:
        lab, leaf = gt.leaf_results(dom, case, goal=proc)
    except Exception:
        r["ok"] = False; r["why"] = "label"; return r
    r.update(ok=True, label=bool(lab), fail_pat=list(validate.fail_pattern(leaf)), case=case, msg_missing=validate.mentions(uk, msg),
             sig=hashlib.md5((json.dumps(db, sort_keys=True) + json.dumps(uk, sort_keys=True)).encode()).hexdigest())
    return r

def main():
    V = variants(); print(len(V), "variants,", len({(v[0], v[1]) for v in V}), "procedures", flush=True)
    with ThreadPoolExecutor(8) as ex: G = list(ex.map(job, V))
    json.dump([{k: g[k] for k in g if k != "text"} for g in G], open(f"{H}/gen_all_meta.json", "w"))
    stats = collections.Counter(); out = collections.defaultdict(dict); rows = []
    for g, (dom, proc, vi, ref, nreal) in zip(G, V):
        try: cases = json.loads(g["text"]); assert isinstance(cases, list)
        except Exception: stats["parse_fail_variants"] += 1; rows.append({"dom": dom, "proc": proc, "vi": vi, "n_gen": 0, "n_valid": 0, "kept": 0}); continue
        val = []; seen = set(); why = collections.Counter()
        for i, c in enumerate(cases):
            stats["cases_generated"] += 1
            r = validate_case(dom, proc, ref, c, stats)
            if not r["ok"]: why[r["why"]] += 1; continue
            if r["sig"] in seen: why["dup"] += 1; continue
            seen.add(r["sig"]); r["idx"] = i; val.append(r)
        perf = [r for r in val if r["label"]]; ref_ = [r for r in val if not r["label"]]
        kept = []
        while len(kept) < 6 and (perf or ref_):
            for pool in (ref_, perf):     # refuse first, then perform, round robin
                if pool and len(kept) < 6: kept.append(pool.pop(0))
        for j, r in enumerate(kept):
            c = r["case"]; c["action_should_succeed"] = r["label"]; c["synthetic"] = True; c["_variant"] = vi
            c["_cid"] = f"{dom}/{proc}/v{vi}s{j}"; out[dom].setdefault(proc, []).append(c)
        stats["valid"] += len(val); stats["kept"] += len(kept)
        rows.append({"dom": dom, "proc": proc, "vi": vi, "nreal": nreal, "n_gen": len(cases), "n_valid": len(val), "kept": len(kept),
                     "kept_perform": sum(r["label"] for r in kept), "invalid_why": dict(why)})
    os.makedirs(f"{H}/synth_all", exist_ok=True)
    for d in DOMS:
        if d in out: json.dump(out[d], open(f"{H}/synth_all/{d}_tasks.json", "w"))
    json.dump(rows, open(f"{H}/gen_all_rows.json", "w"), indent=1); json.dump(dict(stats), open(f"{H}/gen_all_stats.json", "w"))
    print(dict(stats))
    print("procedures with >=1 kept:", len({(r['dom'], r['proc']) for r in rows if r['kept']}))
if __name__ == "__main__": main()
