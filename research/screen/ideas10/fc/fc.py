"""Pre-mortem forecaster (L1): Sonnet 5.5 via headless claude -p (no tools) reads policy, tools and request and lists the
concrete ways a typical 2024-2025 LLM agent would fail on this case, with probabilities, plus P(success).
usage: fc.py test | run"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, hashlib, threading, time, subprocess, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, ZR.SCREEN + "/ideas9/gap")
from gap import inputs
CACHE = HERE / "sonnet_cache.jsonl"; lock = threading.Lock(); cache = {}
if CACHE.exists():
    for l in CACHE.read_text().splitlines():
        d = json.loads(l); cache[d["k"]] = d["r"]
WD = HERE / "work"; stop = {"v": False}; consec = {"n": 0}; stats = {"calls": 0, "cost": 0.0}

P = """You are doing a pre-mortem. A tool-using LLM customer-service agent of the 2024-2025 generation (models such as GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet, Gemini 1.5/2.0, Llama 3 70B, o1-mini) will handle the case below, in a live multi-turn conversation with the customer, calling the listed tools against the company database, which neither you nor the agent can see in advance. The case is graded strictly: the agent must follow the policy exactly (every required check or confirmation, in order, and the right final action or refusal), and the final database state must be correct.

Assume the agent WILL fail. Find the most likely concrete reasons, specific to THIS case, not generic ones.

=== POLICY ===
{policy}
=== END POLICY ===

=== TOOLS ===
{tools}
=== END TOOLS ===

=== CUSTOMER REQUEST OR SCENARIO ===
{request}
=== END ===

Return ONLY a JSON object, no prose before or after:
{{"failure_modes": [{{"what": "one sentence, specific to this case", "p": probability between 0 and 1 that a typical agent of that generation makes this mistake on this case}}],
  "p_success": probability between 0 and 1 that a typical agent of that generation fully succeeds on this case,
  "hardest_part": "one short phrase"}}
List at most 6 failure modes, most likely first."""

def prompt_of(it):
    pol = it[2] if it[0] != "sopbench" else it[2]
    return P.format(policy=pol, tools=it[3], request=it[4])

def call(prompt):
    k = hashlib.sha256(prompt.encode()).hexdigest()
    if k in cache: return cache[k]
    if stop["v"]: return None
    cmd = ["claude", "-p", "--model", "sonnet", "--tools", "", "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
    text = None
    for attempt in range(6):
        try:
            p = subprocess.run(cmd, input=prompt, cwd=WD, capture_output=True, text=True, timeout=600)
            res = json.loads(p.stdout)
        except Exception as e:
            res = {"is_error": True, "result": f"exc {e}"}
        if not res.get("is_error") and res.get("result"):
            text = res["result"]; stats["cost"] += res.get("total_cost_usd") or 0; break
        txt = json.dumps(res)[:800].lower()
        time.sleep(min(300, 30 * (attempt + 1)) if ("limit" in txt or "rate" in txt or "overload" in txt or "429" in txt) else 10)
    with lock:
        stats["calls"] += 1
        if text is not None:
            cache[k] = text; consec["n"] = 0
            with open(CACHE, "a") as f: f.write(json.dumps({"k": k, "r": text}) + "\n")
        else:
            consec["n"] += 1
            if consec["n"] >= 6: stop["v"] = True
        if stats["calls"] % 25 == 0: print("calls", stats["calls"], "cost_eq", round(stats["cost"], 2), flush=True)
    return text

if __name__ == "__main__":
    items = inputs(); mode = sys.argv[1]
    if mode == "test":
        rnd = random.Random(3); items = rnd.sample([i for i in items if i[0] == "tau2"], 2) + rnd.sample([i for i in items if i[0] == "sopbench"], 1)
    order = [i for i in items if i[0] != "sopbench"] + [i for i in items if i[0] == "sopbench"]   # tau2 and banking first
    prompts = {}
    for it in order: prompts.setdefault(prompt_of(it), []).append(it[1])
    ps = list(prompts)
    with ThreadPoolExecutor(int(__import__("os").environ.get("NW", "6"))) as ex: outs = list(ex.map(call, ps))
    res = {c: o for p, o in zip(ps, outs) for c in prompts[p]}
    json.dump(res, open(HERE / ("test_raw.json" if mode == "test" else "fc_raw.json"), "w"), indent=0)
    print("done", len(res), "missing", sum(v is None for v in res.values()), "cost_eq", round(stats["cost"], 2))
    if mode == "test":
        for c, o in res.items(): print(c, o[:1500])
