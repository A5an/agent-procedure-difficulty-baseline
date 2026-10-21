import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import os, sys, json, hashlib, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
S = ZR.SCREEN
PP = S + "/ideas3/paper_pilot"; PT = S + "/ideas3/plan_tools"
POL = ZR.DATA + "/tau2-domains/tasks"
REPO = ZR.REPO
sys.path.insert(0, REPO + "/src")
_cwd = os.getcwd()
import llm as LLM
os.chdir(_cwd)
MODEL = "gemini-3.8-flash"
PERSONAS = [
 ("rushed", "a rushed agent: acts on the first plausible reading of the request, calls the action tool as soon as it seems justified, and only does a check if it is obviously the very first thing to do"),
 ("literal", "a literal agent: does exactly what the text states explicitly and nothing more; it performs a check only when the policy or the tool description spells out that the check is needed right there, and it does not infer implied requirements"),
 ("forgetful", "an agent that forgets checks that are not written next to the action: it remembers conditions stated in the same sentence as the action, but drops conditions stated elsewhere in the policy or tool descriptions"),
 ("average", "an average competent LLM agent: usually follows the policy, sometimes misses one subtle condition or an ordering requirement"),
 ("expert", "a careful expert agent: reads the whole policy, checks every condition before any irreversible action, refuses when any condition fails or cannot be verified"),
]
def tool_lines(dom):
    if dom in ("airline", "retail", "telecom"):
        out = []
        for l in json.load(open(PT + "/tool_text.json"))[dom].split("\n"):
            try: t = json.loads(l)
            except Exception: continue
            out.append(f"- {t['name']}({', '.join((t.get('parameters') or {}).get('properties', {}))}): {t.get('description','')[:300]}")
        return "\n".join(out)
    d = json.load(open(PP + "/domain_tools.json"))[dom]; lines = []
    for tl in d["tools"]:
        f = tl["function"]; props = f["parameters"].get("properties", {})
        lines.append(f"- {f['name']}({', '.join(props)}): {f['description']}")
    lines.append("- exit_conversation(): ends the conversation")
    return "\n".join(lines)
def policy(dom):
    if dom == "telecom": t = open(POL + "/telecom_main_policy.md").read() + "\n\n" + open(POL + "/telecom_tech_support_manual.md").read()
    else: t = open(POL + f"/{dom}_policy.md").read()
    return f"=== DOMAIN POLICY ({dom}) ===\n{t}\n=== END POLICY ==="
def build_prompt(cid, statement):
    dom = cid.split("/")[0]
    if dom in ("airline", "retail", "telecom"):
        body = f"{policy(dom)}\n\nCASE:\n{statement}\n\nTOOLS:\n{tool_lines(dom)}"
    else:
        body = f"CASE:\n{statement.split(chr(10)+'Available tools:')[0]}\n\nTOOLS:\n{tool_lines(dom)}"
    plist = "\n".join(f"{i+1}. {n}: {d}" for i, (n, d) in enumerate(PERSONAS))
    return ("Below is a customer-service case and the tools of the agent. Nothing is executed now: you only simulate, on paper. "
            "Role-play five different LLM agents of graded reliability handling THIS case. The five personas, from least to most reliable:\n"
            + plist +
            "\n\nSimulate each persona realistically, including the mistakes that kind of agent really makes. Do not make every persona correct and do not make them identical: "
            "a weaker persona really omits or reorders checks that a stronger one performs. Assume each lookup returns what is needed to continue. "
            "For each persona write the ordered tool names it would call (tool names exactly as listed, no arguments), its final decision, "
            "and which policy conditions or checks it would skip or never think of.\n\n" + body +
            '\n\nAnswer with JSON only: {"personas": [{"persona": "<name>", "tool_calls": ["<tool>", ...], "final_decision": "perform" | "refuse" | "transfer", "skipped_conditions": ["<short text>", ...]}, ... five entries in the order above]}')
_lock = threading.Lock(); _n = {"n": 0}
def key(prompt, temp): return hashlib.sha256(f"{MODEL}|{temp}|{prompt}".encode()).hexdigest()
def load_cache():
    p = HERE + "/cache.jsonl"; c = {}
    if os.path.exists(p):
        for l in open(p):
            try: r = json.loads(l); c[r["k"]] = r
            except Exception: pass
    return c
def call(prompt, temp, retries=8):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temp, "maxOutputTokens": 8000, "responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "low"}}}
    data = json.dumps(body).encode()
    for a in range(retries):
        url, auth = LLM._endpoint(MODEL)
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth})
        try:
            with urllib.request.urlopen(req, timeout=240) as r: d = json.loads(r.read())
            parts = ((d.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
            return "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except urllib.error.HTTPError as e:
            if e.code == 401: LLM._TOKEN["exp"] = 0.0
            if e.code in (401, 429, 500, 502, 503, 504) and a < retries - 1: time.sleep(min(60, 2 ** a + 1)); continue
            raise RuntimeError(f"HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if a < retries - 1: time.sleep(min(60, 2 ** a + 1)); continue
            raise
def run(items, temp=0, workers=6, limit=None):
    cache = load_cache(); path = HERE + "/cache.jsonl"; todo = {}
    for cid, pr in items.items():
        k = key(pr, temp)
        if k not in cache and k not in todo: todo[k] = pr
    print("unique to call", len(todo), "cached", len(items) - len(todo), flush=True)
    if limit is not None: todo = dict(list(todo.items())[:limit])
    def work(kp):
        k, pr = kp
        try: text = call(pr, temp); err = None
        except Exception as e: text, err = "", str(e)[:100]
        with _lock:
            if not err and text:
                open(path, "a").write(json.dumps({"k": k, "text": text}) + "\n"); cache[k] = {"k": k, "text": text}
            _n["n"] += 1
            if _n["n"] % 50 == 0: print("done", _n["n"], flush=True)
    with ThreadPoolExecutor(workers) as ex: list(ex.map(work, todo.items()))
    return {cid: cache.get(key(pr, temp)) for cid, pr in items.items()}
