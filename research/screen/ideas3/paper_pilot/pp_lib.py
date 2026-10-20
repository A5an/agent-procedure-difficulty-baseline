import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os, sys, json, hashlib, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = ZR.REPO
sys.path.insert(0, REPO + "/src")
_cwd = os.getcwd()
import llm as LLM   # repo client, only used for endpoint and auth
os.chdir(_cwd)

TAU_AGENT_TOOLS = {"telecom": ["get_customer_by_phone", "get_customer_by_id", "get_customer_by_name", "get_details_by_id",
    "get_bills_for_customer", "send_payment_request", "resume_line", "suspend_line", "enable_roaming", "disable_roaming",
    "refuel_data", "get_data_usage", "get_available_plans", "transfer_to_human_agents"]}

def tool_block(case_id):
    dom = case_id.split("/")[0]
    if dom in ("airline", "retail", "telecom"):
        t = json.load(open(HERE + "/tau2_tools.json"))[dom]
        keep = TAU_AGENT_TOOLS.get(dom, list(t))
        keep = [k for k in keep if k in t] or keep
        lines = []
        for n in keep:
            args = t[n]["args"] if n in t else []
            lines.append(f"- {n}({', '.join(args)})")
        if dom == "telecom":
            lines.append("(You can also ask the customer to perform actions on their phone and report the result.)")
        return "\n".join(lines)
    d = json.load(open(HERE + "/domain_tools.json"))[dom]
    lines = []
    for tl in d["tools"]:
        f = tl["function"]; props = f["parameters"].get("properties", {})
        lines.append(f"- {f['name']}({', '.join(props)}): {f['description']}")
    lines.append("- exit_conversation(): ends the conversation")
    return "\n".join(lines)

def build_prompt(case_id, statement):
    dom = case_id.split("/")[0]
    if dom in ("airline", "retail", "telecom"):
        policy = json.load(open(HERE + "/tau2_policy.json"))[dom]
        body = f"AGENT POLICY:\n{policy}\n\nCASE:\n{statement}\n\nTOOLS YOU CAN CALL:\n{tool_block(case_id)}"
    else:
        stmt = statement.split("\nAvailable tools:")[0]
        body = f"CASE:\n{stmt}\n\nTOOLS YOU CAN CALL:\n{tool_block(case_id)}"
    return ("You are the customer service agent described below. Right now you have NO tools and NO system access, so you cannot run anything. "
            "Write on paper the exact plan you would carry out for this case: the ordered list of tool calls you would make, with the arguments you would pass, "
            "and your final decision. Assume each lookup returns what you need to continue. Use only the tool names listed.\n\n"
            + body +
            "\n\nAnswer with JSON only, in this form:\n"
            '{"tool_calls": [{"name": "<tool>", "arguments": {"<arg>": "<value>"}}, ...], '
            '"final_decision": "perform" | "refuse" | "transfer", '
            '"refusal_conditions": ["<condition under which you would refuse instead>", ...]}\n'
            "final_decision is what you expect to do with the customer's request in the end. "
            "Put every tool call you would make, in order, including the call that carries out the requested action if you would make it.")

_lock = threading.Lock(); _count = {"n": 0}
def key(model, prompt): return hashlib.sha256((model + "\x00" + prompt).encode()).hexdigest()

def load_cache(model):
    p = f"{HERE}/cache_{model}.jsonl"; c = {}
    if os.path.exists(p):
        for l in open(p):
            try: r = json.loads(l); c[r["k"]] = r
            except Exception: pass
    return c

def call(model, prompt, retries=8):
    think = {}
    if model == "gemini-3.8-flash": think = {"thinkingConfig": {"thinkingLevel": "low"}}
    elif model == "gemini-2.5-flash": think = {"thinkingConfig": {"thinkingBudget": 1024}}
    elif model == "gemini-2.5-flash-lite": think = {"thinkingConfig": {"thinkingBudget": 0}}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 6000, "responseMimeType": "application/json", **think}}
    data = json.dumps(body).encode()
    for a in range(retries):
        url, auth = LLM._endpoint(model)
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth})
        try:
            with urllib.request.urlopen(req, timeout=180) as r: d = json.loads(r.read())
            cand = (d.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            return text, {k: v for k, v in (d.get("usageMetadata") or {}).items() if "Count" in k} | {"finish": cand.get("finishReason")}
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:200]
            if e.code == 401: LLM._TOKEN["exp"] = 0.0
            if e.code in (401, 429, 500, 502, 503, 504) and a < retries - 1:
                time.sleep(min(60, 2 ** a + 1)); continue
            raise RuntimeError(f"HTTP {e.code}: {msg}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if a < retries - 1: time.sleep(min(60, 2 ** a + 1)); continue
            raise

def run(model, items, workers=8, limit_calls=None):
    """items: {case_id: prompt}. Cached by prompt. Returns {case_id: record}."""
    cache = load_cache(model); path = f"{HERE}/cache_{model}.jsonl"
    todo = {}
    for cid, pr in items.items():
        k = key(model, pr)
        if k not in cache and k not in todo: todo[k] = pr
    print(model, "unique to call", len(todo), "cached", len(items) - len(todo), flush=True)
    if limit_calls is not None: todo = dict(list(todo.items())[:limit_calls])
    def work(kp):
        k, pr = kp
        try: text, u = call(model, pr); err = None
        except Exception as e: text, u, err = "", {}, str(e)[:200]
        rec = {"k": k, "text": text, "usage": u, "err": err}
        with _lock:
            if not err:
                open(path, "a").write(json.dumps(rec) + "\n"); cache[k] = rec
            _count["n"] += 1
            if _count["n"] % 100 == 0: print("done", _count["n"], flush=True)
    with ThreadPoolExecutor(workers) as ex: list(ex.map(work, todo.items()))
    return {cid: cache.get(key(model, pr)) for cid, pr in items.items()}
