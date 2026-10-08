"""One LLM call per policy unit -> 10 synthetic scenarios from the policy text and the database SCHEMA only.
Schema = type skeleton of the domain's shipped default_data (documentation-level example), not of any benchmark case.
Variant F of the earlier emulate agent (every condition fails at least once, some scenarios satisfy all)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, hashlib, copy, time, threading, importlib
sys.path.insert(0, ZR.REPO + "/src")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, ZR.SCREEN + "/ideas2/emulate")
from concurrent.futures import ThreadPoolExecutor
import urllib.request, urllib.error
import llm, sop, common
import gen as G   # only skeleton() is used; its import does not read cases
CACHE = os.path.join(HERE, "cache_synth.jsonl")
_lock = threading.Lock(); _mem = {}; STATS = {"calls": 0}
if os.path.exists(CACHE):
    for l in open(CACHE):
        r = json.loads(l); _mem[r["key"]] = r
_sch = {}
def schema(dom):
    if dom not in _sch:
        mod = importlib.import_module(f"env.domains.{dom}.{dom}")
        _sch[dom] = (G.skeleton(mod.default_data, fmt=True), getattr(mod, "default_data_descriptions", {}))
    return _sch[dom]
def build_prompt(t):
    dom, proc = t["domain"], t["user_goal"]
    instr = sop.agent_view(t)[1]["instructions"]
    sk, desc = schema(dom)
    ukeys = sorted(t["user_known"].keys())
    return f"""You are building test scenarios for a customer-service agent that follows a written policy and has tools that read and change a database.

The agent sees exactly this policy text:
<<<POLICY
{instr}
POLICY>>>

The database schema (field names and value types only, "<key>" stands for an arbitrary identifier such as a username or an id, several entries are allowed under it):
{json.dumps(sk, indent=1)}

Field descriptions: {json.dumps(desc)}

Task. Write 10 different realistic scenarios in which a customer asks the agent to do this one action: `{proc}`.
Each scenario has:
- "initial_database": the full database before the conversation, following the schema exactly (same field names, nesting and value types). Keep it small, 1 to 3 entries per collection, but complete.
- "user_known": the facts the customer knows and gives to the agent. The keys must be exactly these: {ukeys}. Use the credential value the customer would present, it may be right or wrong. Put the values the action needs.
- "user_prompt": the first message of the customer, 1 to 4 sentences, in natural language, which states the request and mentions every value in "user_known".
- "should_perform": true if, given the policy and the database, the agent should carry out `{proc}`, false if the agent must refuse because some condition of the policy is not met. Think about the exact policy conditions that apply to `{proc}`.
Use only values and rules that follow from the policy and the schema. Customers ask for the action in every case, whether or not it is allowed.
Make the scenarios different from each other. Across the 10 scenarios let each condition of the policy for `{proc}` be the one that fails at least once, and let some scenarios have every condition satisfied.
Return only a JSON list of 10 objects with the keys "initial_database", "user_known", "user_prompt", "should_perform".
"""
def call(prompt, rep, temp=0.7):
    key = hashlib.sha256(f"{prompt}|{temp}|{rep}".encode()).hexdigest()
    with _lock:
        if key in _mem: return _mem[key]
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temp, "maxOutputTokens": 40000, "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingLevel": "low"}}}
    data = json.dumps(body).encode()
    for attempt in range(7):
        url, auth = llm._endpoint(llm.MODEL)
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth})
        try:
            with urllib.request.urlopen(req, timeout=300) as r: d = json.loads(r.read())
            cand = (d.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            rec = {"key": key, "text": text, "finish": cand.get("finishReason")}
            with _lock:
                _mem[key] = rec; STATS["calls"] += 1
                open(CACHE, "a").write(json.dumps(rec) + "\n")
            return rec
        except urllib.error.HTTPError as e:
            if e.code == 401: llm._TOKEN["exp"] = 0.0
            if e.code in (401, 429, 500, 502, 503, 504): time.sleep(min(60, 2 ** attempt + 1)); continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError): time.sleep(min(60, 2 ** attempt + 1))
    return {"key": key, "text": "", "finish": "FAIL"}
def units():
    u = {}
    for t in sop.load_cases(): u.setdefault(common.key_of(t), t)
    return u
def valid(t, c):
    import gt
    try:
        db, uk = c["initial_database"], c["user_known"]
        if not (isinstance(db, dict) and isinstance(uk, dict)): return False
        if set(uk.keys()) != set(t["user_known"].keys()): return False
        case = copy.deepcopy(t); case["initial_database"] = db; case["user_known"] = uk
        ok, err = gt.load_check(t["domain"], case, goal=t["user_goal"])
        return ok
    except Exception: return False
def job(a):
    k, t, rep = a
    rec = call(build_prompt(t), rep)
    try: cs = json.loads(rec["text"]); assert isinstance(cs, list)
    except Exception: cs = []
    return k, rep, [c for c in cs if isinstance(c, dict)]
if __name__ == "__main__":
    u = units(); out = {}
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    keys = list(u)[:lim] if lim else list(u)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(job, [(k, u[k], 0) for k in keys]))
    ok = {}
    for k, rep, cs in res: ok[k] = [c for c in cs if valid(u[k], c)]; out[k] = {"n_gen": len(cs)}
    # one retry call for units with fewer than 4 valid scenarios
    low = [k for k in keys if len(ok[k]) < 4]
    with ThreadPoolExecutor(8) as ex:
        for k, rep, cs in ex.map(job, [(k, u[k], 1) for k in low]):
            ok[k] += [c for c in cs if valid(u[k], c)]
    for k in keys: out[k]["scen"] = ok[k][:12]
    json.dump(out, open(os.path.join(HERE, "synthetic.json" if not lim else "synthetic_test.json"), "w"))
    n = [len(ok[k]) for k in keys]
    print("units", len(keys), "calls", STATS, "mean valid", sum(n) / len(n), "units<4", sum(1 for x in n if x < 4), "zero", sum(1 for x in n if x == 0))
