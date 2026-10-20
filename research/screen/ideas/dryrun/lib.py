"""Dry-run feature group: prompts, cached LLM calls (with temperature), parsing."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import hashlib, json, os, subprocess, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(ZR.REPO)
POL = Path(ZR.DATA + "/tau2-domains/tasks")
sys.path.insert(0, str(REPO / "src"))
os.environ["LLM_BACKEND"] = "vertex"
os.environ["GOOGLE_CLOUD_PROJECT"] = subprocess.run(["gcloud", "config", "get-value", "project"],
                                                    capture_output=True, text=True).stdout.strip()
import llm  # noqa

HERE = Path(__file__).parent
CACHE = HERE / "cache" / "calls.jsonl"
LOCK = threading.Lock()
MAXCALLS = 6000
STATE = {"n": 0}
STATUS = ["satisfied_by_request", "violated_by_request", "needs_system_data", "not_applicable"]
DEC = ["perform", "refuse", "transfer_or_other"]
THINK = os.environ.get("DRY_THINK", "low")


def generate_t(prompt, temperature=0.0, max_tokens=1000, thinking="low", model=llm.MODEL, retries=8):
    """Copy of repo llm.generate with a temperature argument."""
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens,
                                 "thinkingConfig": {"thinkingLevel": thinking}}}
    data = json.dumps(body).encode()
    for attempt in range(retries):
        url, auth = llm._endpoint(model)
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            cand = (d.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            return "".join(p.get("text", "") for p in parts if not p.get("thought")), cand.get("finishReason")
        except urllib.error.HTTPError as e:
            if e.code == 401:
                llm._TOKEN["exp"] = 0.0
            if e.code in (401, 429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(min(60, 2 ** attempt + 1)); continue
            raise RuntimeError(f"HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < retries - 1:
                time.sleep(min(60, 2 ** attempt + 1)); continue
            raise


def policy_for(case_id):
    if case_id.startswith("airline/"):
        return (POL / "airline_policy.md").read_text()
    if case_id.startswith("retail/"):
        return (POL / "retail_policy.md").read_text()
    if case_id.startswith("telecom/"):
        return ((POL / "telecom_main_policy.md").read_text() + "\n\n## Tech support workflow\n"
                + (POL / "telecom_tech_support_workflow.md").read_text())
    return None  # SOPBench: policy is inside the statement


DEFS = """Status definitions for each policy condition:
- satisfied_by_request: the customer message itself shows the condition holds (for example the amount is within a stated limit, the required parameter is given).
- violated_by_request: the customer message itself shows the condition fails, so the action must be refused (for example a stated amount exceeds a limit written in the policy, a required value is missing or invalid as written).
- needs_system_data: the message alone does not decide the condition, it depends on records the agent would look up (account balance, whether a user or order exists, order status, stored settings). You have NO access to any database, so use this status whenever the request alone does not decide it. Do not guess.
- not_applicable: the condition is irrelevant for this request."""

FORMAT = """Return exactly one JSON object and nothing else:
{"conditions": [{"condition": "<short text>", "status": "<one of the four statuses>"}],
 "outcome_probs": {"perform": <0..1>, "refuse": <0..1>, "transfer_or_other": <0..1>},
 "n_lookups": <integer, number of system lookups the agent must do before the final action>,
 "n_customer_facts": <integer, number of facts the agent must still obtain from the customer>,
 "ambiguous_request": <0 or 1>,
 "ambiguous_policy": <0 or 1>}
outcome_probs must sum to 1. "perform" = the agent carries out the requested action, "refuse" = the agent must decline, "transfer_or_other" = transfer to a human agent or another resolution. ambiguous_request = 1 if the customer message is unclear or contradictory. ambiguous_policy = 1 if the policy does not clearly decide this case.
Tiny example of the format: for a request "withdraw $500" under a policy "amount must not exceed the balance", answer {"conditions": [{"condition": "amount does not exceed the balance", "status": "needs_system_data"}], "outcome_probs": {"perform": 0.6, "refuse": 0.4, "transfer_or_other": 0.0}, "n_lookups": 1, "n_customer_facts": 0, "ambiguous_request": 0, "ambiguous_policy": 0}"""


def main_prompt(case_id, text):
    pol = policy_for(case_id)
    head = ("You are a compliance officer doing a dry run. Read the customer request next to the policy and judge, before touching "
            "any system, whether the request will have to be refused, which policy conditions cannot be checked without system data, "
            "and how much lookup work the agent will need.\n\n")
    if pol:
        head += f"=== POLICY ===\n{pol}\n\n=== CASE ===\n{text}\n\n"
    else:
        head += f"=== CASE (includes the policy for the requested action) ===\n{text}\n\n"
    return head + DEFS + "\n\n" + FORMAT


def decision_prompt(case_id, text):
    pol = policy_for(case_id)
    head = "You are a customer service agent about to decide how to handle a request. You cannot see any database.\n\n"
    head += (f"=== POLICY ===\n{pol}\n\n=== CASE ===\n{text}\n\n" if pol else f"=== CASE (includes the policy) ===\n{text}\n\n")
    return head + ('Decide the most likely correct final outcome. Answer with exactly one word: perform (carry out the request), '
                   'refuse (decline it), or transfer_or_other (transfer to a human or anything else). No other text.')


def parse_json(t):
    t = t.strip()
    a, b = t.find("{"), t.rfind("}")
    return json.loads(t[a:b + 1])


def validate_main(d):
    try:
        c = d["conditions"]
        if not isinstance(c, list) or any(x.get("status") not in STATUS for x in c):
            return False
        p = d["outcome_probs"]
        v = [float(p[k]) for k in DEC]
        if min(v) < 0 or not (0.8 <= sum(v) <= 1.2):
            return False
        for k in ("n_lookups", "n_customer_facts"):
            if not (0 <= int(d[k]) <= 50):
                return False
        return all(int(d[k]) in (0, 1) for k in ("ambiguous_request", "ambiguous_policy"))
    except Exception:
        return False


def parse_dec(t):
    w = t.strip().lower().replace(".", "").replace("`", "").split()
    if not w:
        return None
    return w[0] if w[0] in DEC else None


_cache = {}
def load_cache():
    if CACHE.exists():
        for l in open(CACHE):
            r = json.loads(l)
            _cache[r["key"]] = r


def one(kind, case_id, text, k=0):
    """kind main | dec (k = sample index 1..4). Cached by hash of prompt + kind + k."""
    prompt = main_prompt(case_id, text) if kind == "main" else decision_prompt(case_id, text)
    temp = 0.0 if kind == "main" else 1.0
    key = hashlib.sha256(f"{kind}|{k}|{THINK}|{prompt}".encode()).hexdigest()[:20]
    if key in _cache:
        return _cache[key]
    rec = {"key": key, "kind": kind, "k": k, "case_id": case_id, "ok": False, "parsed": None}
    for attempt in range(3 if kind == "main" else 2):
        with LOCK:
            if STATE["n"] >= MAXCALLS:
                raise SystemExit("call budget reached")
            STATE["n"] += 1
        try:
            out, fin = generate_t(prompt + ("" if attempt == 0 else "\n\nYour previous answer was not valid. Follow the requested form exactly."),
                                  temperature=temp, max_tokens=6000 if kind == "main" else 1500, thinking=THINK)
            rec["raw"] = out[:600]
            if kind == "main":
                d = parse_json(out)
                if validate_main(d):
                    rec.update(ok=True, parsed=d); break
            else:
                d = parse_dec(out)
                if d:
                    rec.update(ok=True, parsed=d); break
        except SystemExit:
            raise
        except Exception as e:
            rec["raw"] = f"ERR {type(e).__name__}"
    rec["attempts"] = attempt + 1
    with LOCK:
        _cache[key] = rec
        with open(CACHE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def statements():
    out = []
    for b in ("sopbench", "tau2"):
        for l in open(REPO / "data" / b / "statements.jsonl"):
            r = json.loads(l); out.append((b, r["case_id"], r["text"]))
    return out


def run(items, kinds=("main", "dec"), workers=8):
    jobs = []
    for b, cid, text in items:
        if "main" in kinds:
            jobs.append(("main", cid, text, 0))
        if "dec" in kinds:
            jobs += [("dec", cid, text, k) for k in range(1, 5)]
    n = 0
    with ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(lambda j: one(*j), jobs):
            n += 1
            if n % 200 == 0:
                print(n, "of", len(jobs), "calls this process", STATE["n"], flush=True)
    print("done; LLM calls this process", STATE["n"])


load_cache()
