"""Step 2: one gemini-3.8-flash call per (procedure, variant, replicate) -> 10 synthetic cases as JSON. Cached by prompt hash."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, hashlib, copy, time, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, ZR.REPO + "/src")
import gt, llm, urllib.request, urllib.error
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache_gen.jsonl")
SEL = [("bank","pay_loan"),("bank","apply_credit_card"),("dmv","schedule_test"),("dmv","renew_vehicle"),
       ("healthcare","schedule_appointment"),("healthcare","submit_claim"),("hotel","book_room"),
       ("library","borrow_book"),("online_market","return_order"),("university","enroll_course")]
_lock = threading.Lock(); _calls = [0]

def _idlike(k):
    return any(ch.isdigit() or ch.isupper() or ch in ' -' for ch in k)

def skeleton(v, fmt=False, coll=None, path=""):
    """Type skeleton of a database value. A dict is shown as {"<key>": ...} (a collection) when it has 2 or more entries with
    the same skeleton, or when its path is in `coll` (key set differs between real cases of the domain). fmt=True adds a
    format example to string values that contain digits (dates, times, ids)."""
    coll = coll or set()
    if isinstance(v, bool): return "bool"
    if isinstance(v, int): return "int"
    if isinstance(v, float): return "float"
    if isinstance(v, str):
        if fmt and any(ch.isdigit() for ch in v) and len(v) <= 30: return f'str, format like "{v}"'
        return "str"
    if v is None: return "null"
    if isinstance(v, list): return [skeleton(v[0], fmt, coll, path + "[]")] if v else []
    if isinstance(v, dict):
        isc = path in coll
        subs = {k: skeleton(x, fmt, coll, path + ("/*" if isc else "/" + k)) for k, x in v.items()}
        vals = list(v.values())
        scal = [isinstance(x, (str, int, float, bool)) or x is None for x in vals]
        uniform = len({json.dumps(sk, sort_keys=True) for sk in subs.values()}) == 1 if subs else False
        mixed_scalar = any(scal) and not all(scal) or (all(scal) and not uniform)   # a record with differently typed fields
        if v and not mixed_scalar and (isc or any(_idlike(k) for k in v) or (len(v) >= 2 and not all(scal) and uniform)):
            return {"<key>": next(iter(subs.values()))}
        return subs
    return type(v).__name__

_coll = {}
def collections_for(dom):
    """paths of dicts whose key set varies between the real cases of the domain (collections such as accounts, not fixed fields).
    Iterates because children of a collection are addressed by "*"."""
    if dom in _coll: return _coll[dom]
    dbs = [c["initial_database"] for cs in json.load(open(f"{gt.SOP}/data/{dom}_tasks.json")).values() for c in cs]
    coll = set()
    for _ in range(5):
        seen = {}
        def walk(v, path):
            if isinstance(v, dict):
                seen.setdefault(path, set()).add(frozenset(v.keys()))
                for k, x in v.items(): walk(x, path + ("/*" if path in coll else "/" + k))
            elif isinstance(v, list):
                for x in v[:1]: walk(x, path + "[]")
        for d in dbs: walk(d, "")
        new = {p for p, ks in seen.items() if len(ks) > 1} | coll
        if new == coll: break
        coll = new
    _coll[dom] = coll
    return coll

def merge(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items(): out[k] = merge(a[k], v) if k in a else v
        return out
    if isinstance(a, list) and isinstance(b, list):
        if not a: return b
        if not b: return a
        return [merge(a[0], b[0])]
    if a == "null": return b
    return a

_msk = {}
def domain_schema(dom, fmt):
    """schema of the whole domain database: union of the skeletons of every real case database of the domain (what a schema document would list)"""
    if (dom, fmt) not in _msk:
        coll = collections_for(dom); out = None
        for cs in json.load(open(f"{gt.SOP}/data/{dom}_tasks.json")).values():
            for c in cs:
                sk = skeleton(c["initial_database"], fmt=fmt, coll=coll)
                out = sk if out is None else merge(out, sk)
        _msk[(dom, fmt)] = out
    return _msk[(dom, fmt)]

def modal_variant(dom, proc):
    cs = json.load(open(f"{gt.SOP}/data/{dom}_tasks.json"))[proc]
    cnt = {}
    for c in cs: cnt.setdefault(json.dumps(c["constraints_original"], sort_keys=True), []).append(c)
    best = max(cnt.values(), key=lambda L: (len(L), len(json.dumps(L[0]["constraints_original"]))))
    return cs, best

def descriptions(dom):
    import importlib
    mod = {"bank":"env.domains.bank.bank","dmv":"env.domains.dmv.dmv","healthcare":"env.domains.healthcare.healthcare",
           "hotel":"env.domains.hotel.hotel","library":"env.domains.library.library",
           "online_market":"env.domains.online_market.online_market","university":"env.domains.university.university"}[dom]
    m = importlib.import_module(mod)
    return getattr(m, "default_data_descriptions", {})

def build_prompt(dom, proc, variant):
    cs, same = modal_variant(dom, proc)
    ref = same[0]
    instr, tools = gt.policy_prompt(dom, ref, goal=proc)
    schema = domain_schema(dom, True) if variant in ('E','F') else skeleton(ref["initial_database"], fmt=(variant in ('C','D')), coll=collections_for(dom))
    desc = descriptions(dom)
    ukeys = sorted(ref["user_known"].keys())
    base = f"""You are building test scenarios for a customer-service agent that follows a written policy and has tools that read and change a database.

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
"""
    if variant in ("A", "C", "E"):
        base += "Make the scenarios different from each other.\n"
    else:
        base += "Make the scenarios different from each other. Across the 10 scenarios let each condition of the policy for `%s` be the one that fails at least once, and let some scenarios have every condition satisfied.\n" % proc
    base += 'Return only a JSON list of 10 objects with the keys "initial_database", "user_known", "user_prompt", "should_perform".'
    return base, ref, same

def call(prompt, temperature, rep):
    key = hashlib.sha256(f"{prompt}|{temperature}|{rep}".encode()).hexdigest()
    with _lock:
        if os.path.exists(CACHE):
            for line in open(CACHE):
                r = json.loads(line)
                if r["key"] == key: return r
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 40000, "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingLevel": "low"}}}
    data = json.dumps(body).encode()
    for attempt in range(6):
        url, auth = llm._endpoint(llm.MODEL)
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth})
        try:
            with urllib.request.urlopen(req, timeout=300) as r: d = json.loads(r.read())
            cand = (d.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            rec = {"key": key, "text": text, "usage": d.get("usageMetadata"), "finish": cand.get("finishReason")}
            with _lock:
                _calls[0] += 1
                open(CACHE, "a").write(json.dumps(rec) + "\n")
            return rec
        except urllib.error.HTTPError as e:
            if e.code == 401: llm._TOKEN["exp"] = 0.0
            if e.code in (401, 429, 500, 502, 503, 504): time.sleep(2 ** attempt + 1); continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError): time.sleep(2 ** attempt + 1)
    raise RuntimeError("failed")

def job(a):
    dom, proc, variant, rep, temp = a
    prompt, ref, same = build_prompt(dom, proc, variant)
    rec = call(prompt, temp, rep)
    return {"dom": dom, "proc": proc, "variant": variant, "rep": rep, "temp": temp, "key": rec["key"], "finish": rec["finish"],
            "usage": rec["usage"], "text": rec["text"]}

if __name__ == "__main__":
    plan = sys.argv[1]  # "pilot", "full" (v1, first schema skeleton, kept as gen_full.json) or "v2"
    jobs = []
    if plan == "v2": plan_jobs = True
    if plan == "pilot":   # smoke test on 2 procedures
        jobs = [(d, p, "A", 0, 0.7) for d, p in SEL[:2]]
    else:
        for d, p in SEL:
            for v in (("A", "C", "D") if plan == "v2" else ("E", "F")):
                for rep in range(2): jobs.append((d, p, v, rep, 0.7))
    out = []
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(job, jobs): out.append(r); print(r["dom"], r["proc"], r["variant"], r["rep"], r["finish"], r["usage"].get("candidatesTokenCount"), flush=True)
    json.dump(out, open(os.path.join(HERE, f"gen_{plan}.json"), "w"))
