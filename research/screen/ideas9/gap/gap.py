"""Data-gap labelling (L1): one gemini-3.8-flash call per case. Identical prompt template for every benchmark.
usage: gap.py test | run"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, hashlib, threading, time, re, importlib, urllib.request, urllib.error, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
REPO = Path(ZR.REPO)
sys.path.insert(0, str(REPO / "src"))
import llm
HERE = Path(__file__).parent
EXT = Path(ZR.DATA)
POL = EXT / "tau2-domains/tasks"; BANK = EXT / "tau2-banking"; SOP = EXT / "SOPBench-d2622008"
TT = json.load(open(HERE.parent.parent / "ideas3/plan_tools/tool_text.json"))
CACHE = HERE / "llm_cache.jsonl"; lock = threading.Lock(); cache = {}
if CACHE.exists():
    for l in CACHE.read_text().splitlines():
        d = json.loads(l); cache[d["k"]] = d["r"]
ncalls = 0

def call(prompt, tag="gap"):
    global ncalls
    k = hashlib.sha256(f"0|{tag}|{prompt}".encode()).hexdigest()
    if k in cache: return cache[k]
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 6000, "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingLevel": "low"}}}
    data = json.dumps(body).encode(); text = None
    for attempt in range(8):
        url, auth = llm._endpoint(llm.MODEL)
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth})
        try:
            with urllib.request.urlopen(req, timeout=240) as r: d = json.loads(r.read())
            cand = (d.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")); break
        except urllib.error.HTTPError as e:
            if e.code == 401: llm._TOKEN["exp"] = 0.0
            if e.code in (401, 429, 500, 502, 503, 504): time.sleep(min(60, 2 ** attempt + 1)); continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError): time.sleep(min(60, 2 ** attempt + 1))
    with lock:
        ncalls += 1
        if text:
            cache[k] = text
            with open(CACHE, "a") as f: f.write(json.dumps({"k": k, "r": text}) + "\n")
    return text

INSTR = """You are auditing, on paper, where the data comes from for one customer case, before any agent runs. You have no database access and you call nothing. Use only the policy, the tool list and the customer request below.

=== POLICY ===
{policy}
=== END POLICY ===

=== TOOLS (name, description, parameters) ===
{tools}
=== END TOOLS ===

=== CUSTOMER REQUEST ===
{request}
=== END REQUEST ===

Think about the tool calls that correct handling of this request needs: the final action the customer wants (if the policy allows it) and every check or lookup tool that the policy requires before it. Then label the source of every value, using these labels:
GIVEN = the request states the value verbatim.
DERIVABLE = not stated, but computable from what the request states (say how in "how").
FETCH = must be obtained from the system by calling a named tool (name it in "how").
ASK = must be asked from the customer in conversation, it is not in the request and no tool returns it.
MISSING = no source at all for the value.
For conditions use GIVEN, FETCH, ASK or UNKNOWN (UNKNOWN = the source of the input values cannot be determined).

Return ONLY JSON with this schema:
{{
 "arguments": [ {{"tool": "tool_name", "role": "target" or "check", "argument": "required argument name", "source": "GIVEN" or "DERIVABLE" or "FETCH" or "ASK" or "MISSING", "how": "short"}} ],   // every required argument of every tool call the case needs, one row per argument per call
 "conditions": [ {{"condition": "short text of one gating condition of the policy that applies to this request", "source": "GIVEN" or "FETCH" or "ASK" or "UNKNOWN", "how": "short"}} ]   // every gating condition (a test whose result decides whether the action may proceed) that applies to this request, with the source of the input values the test needs. If all inputs of a condition are in the request, use GIVEN. If any input must be looked up, use FETCH.
}}"""

# ---------------- inputs per benchmark ----------------
def tau_policy(dom):
    if dom == "telecom":
        return (POL / "telecom_main_policy.md").read_text() + "\n\n" + (POL / "telecom_tech_support_manual.md").read_text()
    return (POL / f"{dom}_policy.md").read_text()

_sop_tools = {}
def sop_tools(dom):
    if dom not in _sop_tools:
        sys.path.insert(0, str(SOP))
        m = importlib.import_module(f"env.domains.{dom}.{dom}_assistant")
        _sop_tools[dom] = {a["name"]: {"name": a["name"], "description": a.get("description", ""), "parameters": a["parameters"]} for a in m.actions}
    return _sop_tools[dom]

def inputs():
    out = []
    for l in open(REPO / "data/sopbench/statements.jsonl"):
        s = json.loads(l); t = s["text"]; dom = s["case_id"].split("/")[0]
        head, rest = t.split("Policy for this action:\n", 1)
        pol, rest = rest.split("\nCustomer message: ", 1); req, tl = rest.split("\nAvailable tools: ", 1)
        action = re.search(r"Requested action: (\S+)", head).group(1)
        names = [x.strip() for x in tl.split(",")]; T = sop_tools(dom)
        tools = "\n".join(json.dumps(T[n], separators=(",", ":")) for n in names if n in T)
        out.append(("sopbench", s["case_id"], head.strip() + "\n\nPolicy for the requested action:\n" + pol.strip(), tools, req.strip()))
    for l in open(REPO / "data/tau2/statements.jsonl"):
        s = json.loads(l); dom = s["case_id"].split("/")[0]
        out.append(("tau2", s["case_id"], tau_policy(dom), TT[dom], s["text"]))
    bpol = (BANK / "prompts/components/policy_header.md").read_text() + "\n\n" + (BANK / "prompts/components/additional_instructions.md").read_text()
    btools = ("(The knowledge base of this domain, about 700 documents, is NOT included here because it is too long. The agent's tools are listed by name only.)\n"
              "KB_search_bm25(query, k), KB_search_dense(query, k), shell(command), get_current_time(), transfer_to_human_agents(summary), "
              "give_discoverable_user_tool(discoverable_tool_name), unlock_discoverable_agent_tool(agent_tool_name), "
              "call_discoverable_agent_tool(agent_tool_name, arguments), and read and write tools on the customer database whose schemas are not shown.")
    for l in open(REPO / "data/tauk_banking/statements.jsonl"):
        s = json.loads(l); out.append(("banking", s["case_id"], bpol, btools, s["text"]))
    return out

def prompt_of(it): return INSTR.format(policy=it[2], tools=it[3], request=it[4])

if __name__ == "__main__":
    mode = sys.argv[1]; items = inputs()
    print("items", len(items), "distinct prompts", len({prompt_of(i) for i in items}))
    if mode == "test":
        rnd = random.Random(1)
        sel = rnd.sample([i for i in items if i[0] == "sopbench"], 4) + rnd.sample([i for i in items if i[0] == "tau2"], 4) + rnd.sample([i for i in items if i[0] == "banking"], 2)
        with ThreadPoolExecutor(6) as ex: outs = list(ex.map(lambda i: call(prompt_of(i)), sel))
        json.dump({i[1]: o for i, o in zip(sel, outs)}, open(HERE / "test10.json", "w"), indent=1)
        print("calls", ncalls)
    else:
        prompts = {}
        for i in items: prompts.setdefault(prompt_of(i), []).append(i[1])
        ks = list(prompts)
        with ThreadPoolExecutor(6) as ex: outs = list(ex.map(call, ks))
        res = {}
        for p, o in zip(ks, outs):
            for c in prompts[p]: res[c] = o
        json.dump(res, open(HERE / "gap_raw.json", "w")); print("calls", ncalls, "missing", sum(v is None for v in res.values()))
