"""FunSearch / AlphaEvolve style search for case features. usage: evolve.py [n_rounds] [per_island]. Needs Vertex env (see BRIEF).
The proposer sees: record description + truncated search-split example records, parents' code and scores, idea lists. Never labels or evaluator code."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, time, hashlib, re, random, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, ZR.SCREEN + "/ideas")
REPO = Path(ZR.REPO); sys.path.insert(0, str(REPO / "src"))
import llm
import evaluator as EV
from cpulock import cpu_lock
N_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 14
MAX_CALLS = 590
CACHE = HERE / "proposer_cache.jsonl"; ARCH = HERE / "archive.jsonl"; STATE = HERE / "calls_used.txt"
lock = threading.Lock(); cache = {}
if CACHE.exists():
    for l in CACHE.read_text().splitlines(): d = json.loads(l); cache[d["k"]] = d["r"]
calls_used = int(STATE.read_text()) if STATE.exists() else 0

def gen(prompt, tag):
    global calls_used
    k = hashlib.sha256(f"{tag}|{prompt}".encode()).hexdigest()
    if k in cache: return cache[k]
    with lock:
        if calls_used >= MAX_CALLS: return None
        calls_used += 1; STATE.write_text(str(calls_used))
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 1.0, "maxOutputTokens": 6000, "thinkingConfig": {"thinkingLevel": "low"}}}
    data = json.dumps(body).encode(); text = None
    for attempt in range(6):
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
            text = ""; break
        except (urllib.error.URLError, TimeoutError, OSError): time.sleep(min(60, 2 ** attempt + 1))
    if text is None: text = ""
    with lock:
        cache[k] = text
        with open(CACHE, "a") as f: f.write(json.dumps({"k": k, "r": text}) + "\n")
    return text

SCHEMA = """Each case record is a Python dict (a fresh deep copy is passed to your function) with these keys:
  id, benchmark ("sopbench" or "tau2"), domain, procedure_id, procedure (str)
  domain_description (str), policy_text (str: for sopbench the policy of the requested action as bullet conditions; for tau2 the whole domain policy markdown, shared by all cases of the domain)
  message (str: sopbench customer message; tau2 customer scenario text given to a user simulator)
  tools (list of dicts {name, description, parameters (JSON schema dict)}: the agent's tools; sopbench: tools of the domain, tau2: domain tools)
  program (str or None: sopbench only, a Python program an LLM compiled from the policy, calls tools.<name>(...) and returns when refusing)
  plans (list of dicts, 1 for sopbench, up to 3 sampled plans for tau2): {plan: [{tool, purpose, writes_db}], policy_rules: [str], facts_to_ask_customer: [str], outcome: str, ambiguity: int}. LLM-written on paper from documentation, no environment.
  dryrun (dict): {parsed: None or {conditions: [{condition, status}] (status among satisfied_by_request, needs_system_data, violated_by_request, ...), outcome_probs: {perform, refuse, transfer_or_other}, n_lookups, n_customer_facts, ambiguous_request, ambiguous_policy}, plus float columns p_refuse, p_perform, n_conditions, n_violated_by_request, n_needs_system_data, n_lookups, n_customer_facts, ambiguous_request, ambiguous_policy, decision_entropy, refuse_share_sampled (values may be None)}
  conditions (list; sopbench only, [] for tau2): [{text, step_type, step_name, step_tools (list), is_or (0/1), skip_if_must_perform_pred (float), skip_if_must_refuse_pred (float)}] the policy cut into single checks; the two floats are a model's predicted probability that an LLM agent skips that check
  skip_model_case_p_success (float or None; sopbench only): the skip model's predicted success probability for the case
  scenario (dict or None; tau2 only): {n_goals, n_mind_changes, withholds_info, policy_forbidden_request, pressure, n_conditional, needs_computation, ambiguous_reference, n_constraints, n_clauses, policy_clauses_touched: [str]}
Some fields can be None or empty (check before use). The strings can be long (policy up to 17k chars for tau2 telecom)."""

PRIOR_FEATURES = """Already available to the predictor you must ADD to (do not just re-derive them): counts of step types in the policy (lookup, check, existence, authentication, datetime, relation, ...), depth and number of refusal conditions of the procedure (LLM rubric), 18 ADeLe demand levels (LLM rated), counts from the plan (number of calls, writes, reads, distinct tools, rules, facts to ask, refuse flag, ambiguity), compiled program length and number of branches (sopbench), customer-scenario flags (tau2).
Ideas known to FAIL in earlier rounds: embeddings of the text, a zero-shot guess of perform/refuse from the message alone (AUC 0.56 to 0.64), pairwise comparison scores, bag of words, step types alone, plain text length, FMEA style ratings, per-clause difficulty memorisation.
Ideas that SUCCEEDED a little: plan with the real tool list (number of writes and calls), complexity of the compiled code (branches), customer scenario complexity on tau2 (mind changes, withheld info, policy-forbidden request, conditionals), and the skip-model prediction per policy condition (checks like datetime and relation are skipped more often by agents: authentication 88% reliable, datetime and availability 42%, relation 46%).
Facts: on SOPBench most within-domain difficulty comes from whether the case must be performed or refused, which depends on a hidden database; you only see the message, so features that read WHICH checks the message can or cannot settle, and how checks interact with message values, are the most promising. On tau2 each task is its own procedure, a scenario for a user simulator, and difficulty rises with the number of required write actions, goals, constraints, and conditionals."""

THEMES = {
 "policy": "policy structure: the shape of the policy text, the compiled program, the conditions and the tool schemas (nesting, number and type of checks, thresholds, date arithmetic, negations, OR groups, cross references between conditions, order dependence, data flow between tool calls, argument complexity of tools, program control flow)",
 "msgpol": "message versus policy: how the customer message or scenario relates to the policy conditions (which conditions the message settles, numbers in the message compared with thresholds in the policy, fields the policy needs but the message does not give, overlap or conflict between message entities and policy terms, scenario traps, requests near a boundary, multi-goal requests)",
 "plantools": "plan and tools: the LLM plans, the dry run and the skip model (length, writes, order of reads and writes, agreement between sampled plans or between plan and compiled program, dry-run status mix of conditions, entropy, expected number of skipped checks from the skip-model floats, tool-schema difficulty of the planned tools)",
}
ANGLES = ["ratios and normalised counts instead of raw counts", "interaction between two sources (for example plan versus program)", "the single most difficult check in the case", "positions of checks (early or late, before or after the action)", "numbers: thresholds, amounts, dates, and their arithmetic", "ambiguity and vagueness markers", "how many distinct entities or identifiers must be carried through the interaction", "disagreement between different LLM-written descriptions of the same case", "a worst-case or max-based aggregate rather than a mean", "tool schema complexity of tools that the case needs", "counts of negations, exceptions, or 'unless' style clauses", "features that behave differently for sopbench and tau2 (return 0.0 for the benchmark where they do not apply)", "something unusual that you think nobody tried", "conditions the message explicitly satisfies versus conditions that need a database look-up"]

def short(o, n=240):
    if isinstance(o, str): return o[:n] + ("..." if len(o) > n else "")
    if isinstance(o, list): return [short(x, n) for x in o[:3]] + (["...(%d items)" % len(o)] if len(o) > 3 else [])
    if isinstance(o, dict): return {k: short(v, n) for k, v in list(o.items())[:14]}
    return o
def examples():
    out = []; seen = set()
    for l in open(HERE / "records_search.jsonl"):
        r = json.loads(l)
        if r["benchmark"] not in seen and (r["benchmark"] == "tau2" or r["domain"] == "hotel"):
            seen.add(r["benchmark"]); r["policy_text"] = r["policy_text"][:400] + "..."; out.append(json.dumps(short(r, 260))[:3800])
    return "\n".join("Example record (truncated): " + o for o in out)
EXAMPLES = examples()

def build_prompt(island, parents, mode, hist, angle):
    p = f"""You are part of an automated search for numeric features of business-process cases. Each case is one customer request handled by an LLM agent under a written policy (SOPBench: bank, dmv, healthcare, hotel domains; tau2: airline, retail). The aim is to find features that predict which cases are harder for LLM agents than other cases of the SAME domain, in addition to what a strong existing predictor already captures. You cannot see outcomes; you only write code that reads the case record.

{SCHEMA}

{EXAMPLES}

{PRIOR_FEATURES}

TASK: write a Python function `features(record) -> dict` returning up to 8 named float features (finite floats, short names). Rules: at most 70 lines in total INCLUDING blank lines and comments (longer code is rejected unseen, so be compact), pure code, no I/O, no randomness, only imports from math, re, json, statistics, collections, itertools, string. Must run in under 1 second per record and must not crash when a field is None or empty. Features must vary between cases of the same domain (a feature constant within a domain is useless). Never rely on a field that is not in the schema above. The code is scored automatically: the score is the gain in within-domain rank correlation with real agent difficulty when your features are added to the strong predictor, measured by cross-validation across procedures. Pure noise scores about -0.005 (sd 0.005); +0.01 or more is interesting; larger than +0.03 would be exceptional. Features that are mere copies of the existing predictor or just text length are rejected or score zero.

Island theme: {THEMES[island]}.
"""
    if parents:
        p += "\nBest candidates so far on this island (code and score):\n"
        for c in parents: p += f"--- score {c['score']:+.4f}, features {c['feature_names']}\n{c['code']}\n"
    if hist["good"]: p += "\nIdeas from this search that scored well (score, idea): " + "; ".join(f"{s:+.3f} {i}" for s, i in hist["good"]) + "\n"
    if hist["bad"]: p += "\nIdeas from this search that scored badly or were rejected (score or reason, idea): " + "; ".join(f"{s} {i}" for s, i in hist["bad"]) + "\n"
    if mode == "mutate" and parents:
        p += f"\nYour job: write an IMPROVED variant of the first candidate above: keep what seems to work, replace or add features, fix weaknesses, and try this twist if it fits: {angle}.\n"
    else:
        p += f"\nYour job: write a NEW candidate with a genuinely different idea inside the island theme. Angle to consider: {angle}.\n"
    p += "\nAnswer with exactly one ```python code block. Its first line must be a comment `# idea: <one sentence in plain words>`, then the function features(record). No text outside the block."
    return p

def parse(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text or "", re.S)
    if not m: return None, ""
    code = m.group(1).strip("\n")
    i = re.search(r"#\s*idea:\s*(.*)", code)
    return code, (i.group(1).strip() if i else "")

def corr_mat(M, dom):
    Z = EV.demean(M, dom); sd = Z.std(0); sd[sd < 1e-9] = 1; return Z / sd

def main():
    random.seed(7); rng = random.Random(11)
    arch = [json.loads(l) for l in open(ARCH)] if ARCH.exists() else []
    done_rounds = max([a["round"] for a in arch], default=-1) + 1
    islands = list(THEMES)
    mats = {}  # id -> demeaned standardized matrix (search split)
    ok = lambda isl: sorted([a for a in arch if a["island"] == isl and a["status"] == "ok" and a["score"] is not None], key=lambda a: -a["score"])
    nid = len(arch)
    for rnd in range(done_rounds, N_ROUNDS):
        t0 = time.time(); jobs = []
        for isl in islands:
            pop = ok(isl)[:8]
            other = [a for i2 in islands if i2 != isl for a in ok(i2)[:1]]
            for j in range(PER):
                parents = []; mode = "new"
                if pop and rnd >= 1 and rng.random() < 0.7:
                    mode = "mutate"
                    w = np.array([1.0 / (k + 1) for k in range(min(4, len(pop)))]); first = pop[int(np.searchsorted(np.cumsum(w / w.sum()), rng.random()))] if len(w) else pop[0]
                    parents = [first]
                    if len(pop) > 1:  # second parent: most novel among the rest
                        rest = [a for a in pop if a is not first and a["id"] in mats and first["id"] in mats]
                        if rest:
                            nv = lambda a: np.abs(np.corrcoef(mats[a["id"]].T, mats[first["id"]].T)[:mats[a["id"]].shape[1], mats[a["id"]].shape[1]:]).max()
                            parents.append(min(rest, key=nv))
                    if other and rng.random() < 0.25: parents.append(rng.choice(other))
                good = sorted([a for a in arch if a["island"] == isl and a["status"] == "ok" and a["score"] is not None], key=lambda a: -a["score"])
                bad = [a for a in arch if a["island"] == isl and a["idea"] and (a["status"] != "ok" or (a["score"] is not None and a["score"] < -0.01))]
                hist = {"good": [(a["score"], a["idea"]) for a in good[:6] if a["idea"]],
                        "bad": [((f"{a['score']:+.3f}" if a["status"] == "ok" else "rejected: " + str(a.get("reason"))[:60]), a["idea"]) for a in rng.sample(bad, min(6, len(bad)))]}
                jobs.append((isl, j, mode, parents, hist, rng.choice(ANGLES)))
        def run(job):
            isl, j, mode, parents, hist, angle = job
            pr = build_prompt(isl, parents, mode, hist, angle)
            return job, gen(pr, f"r{rnd}|{isl}|{j}")
        with ThreadPoolExecutor(6) as ex: res = list(ex.map(run, jobs))
        best_round = {}
        for job, text in res:
            isl, j, mode, parents, hist, angle = job
            if text is None: continue
            code, idea = parse(text)
            rec = {"id": nid, "round": rnd, "island": isl, "mode": mode, "parents": [p["id"] for p in parents], "idea": idea, "code": code, "calls_used": calls_used}
            nid += 1
            if not code: rec.update(status="reject", reason="no code block", score=None)
            else:
                o = EV.score(code, "search", return_matrix=True)  # tiny CPU job (ridge on 700 rows), run niced without the shared lock
                mat = o.pop("matrix", None); dom = o.pop("domains", None); o.pop("ids", None)
                rec.update(o)
                if mat is not None:
                    Z = corr_mat(mat, dom); mats[rec["id"]] = Z
                    others = [m for k, m in mats.items() if k != rec["id"] and arch and any(a["id"] == k and a["island"] == isl and a["status"] == "ok" for a in arch)]
                    if others:
                        rec["novelty_maxcorr"] = float(max(np.abs(np.corrcoef(Z.T, m.T)[:Z.shape[1], Z.shape[1]:]).max() for m in sorted(others, key=lambda m: 0)[:30]))
            arch.append(rec)
            with open(ARCH, "a") as f: f.write(json.dumps(rec) + "\n")
        sc = [a["score"] for a in arch if a["round"] == rnd and a["score"] is not None]
        print(f"round {rnd} done in {time.time()-t0:.0f}s calls {calls_used} ok {len(sc)}/{len(jobs)} best {max(sc) if sc else None} "
              + " ".join(f"{i}:{(ok(i)[0]['score'] if ok(i) else float('nan')):+.4f}" for i in islands), flush=True)
        if calls_used >= MAX_CALLS: print("call budget reached"); break
if __name__ == "__main__": main()
