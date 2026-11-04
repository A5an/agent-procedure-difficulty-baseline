"""Generate procedure specs with Claude models through headless Claude Code (claude -p), then check them (PREREG.md).

  python gen.py <model: haiku|sonnet|opus> [procedure ids ...] [--workers N]

Same prompt for every model: SPEC_FORMAT.md, the worked example, the brief. No tools, no CLAUDE.md (--setting-sources ""),
no session. The program checks every spec (engine.validate); on errors the model gets them back with its previous JSON,
at most 2 repair rounds. Final spec in gens/<model>/<procedure>.json, every attempt in gens/<model>/<procedure>.log.jsonl.
"""
import json, os, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import engine

HERE = Path(__file__).resolve().parent
FMT = (HERE / "SPEC_FORMAT.md").read_text()
EXAMPLE = (HERE / "example_procedure.json").read_text()
V = os.environ.get("SYNTH_V", "")  # "" = first round, "v2" = hard round (PREREG_V2.md)
BRIEFS = {b["procedure_id"]: b for b in json.load(open(HERE / (f"briefs_{V}.json" if V else "briefs.json")))}
SYS = ("You design realistic business procedures and test cases for evaluating customer-service AI agents. "
       "You follow the specification format exactly and output only one JSON object.")
OUT = HERE / os.environ.get("SYNTH_GEN_DIR", f"gens_{V}" if V else "gens")  # gens_pilot for the pilot
MAX_REPAIRS = 2
MIN_CASES = 6  # a procedure is kept if its procedure-level checks pass and at least this many of its 8 cases pass


def brief_text(b):
    st = b["structure"]
    lines = [f"procedure_id: {b['procedure_id']}", f"domain: {b['domain']}", f"today: {b['today']}",
             f"number of rules: exactly {st['n_rules']}",
             f"main perform outcome: exactly {st['n_calls']} write call(s), in order",
             "escalation branch: yes, a rule sends some requests to a second perform outcome (for example a review or "
             "approval request) instead of the main action" if st["escalation"] else "escalation branch: no",
             "identity verification: yes, a rule compares something the customer states (a claim) with the database"
             if st["verification"] else "identity verification: no claims are needed (claims may be {})",
             ]
    if st.get("min_hidden_rules"):
        lines.append(f"hidden checks: at least {st['min_hidden_rules']} rules check a record other than the main one (a record "
                     "the customer does not talk about, such as the account behind an order, a vehicle's next booking or a catalog "
                     "entry), so the agent has to look it up on its own")
    if st.get("computed_arg"):
        lines.append("computed argument: the main write call takes an argument the agent must compute from at least two values "
                     "(for example a prorated amount, a fee with a cap, or a new date from a date and a number of days); the "
                     "policy explains how to compute it")
    if st.get("db_count"):
        lines.append("counting rule: one rule counts related records with db_count (for example earlier requests of this "
                     "customer), and a search tool lets the agent list them")
    if st.get("note"):
        lines.append(st["note"])
    lines.append("cases (write exactly these, with these ids):")
    lines += [f"  {c['case_id']}: information level {c['level']}, behaviour {c['behavior']}, class {c['cls']}" for c in b["cases"]]
    if any(c["behavior"] == "negotiates" for c in b["cases"]):
        lines.append("behaviour negotiates (not in the format above): the customer first asks for initial_request_values, which "
                     "the rules refuse; only after the agent refuses, they ask for request_values instead (the class above is the "
                     "class of that final request). Put initial_request_values in the case, as for changes_mind.")
    return "\n".join(lines)


def prompt(b, prev=None, errs=None):
    p = (f"{FMT}\n\n## Worked example (another domain; do not copy its structure, entities or wording)\n\n{EXAMPLE}\n\n"
         f"## Your brief\n\n{brief_text(b)}\n\nWrite the complete spec for this brief. Output only the JSON object.")
    if prev is not None:
        p += ("\n\n## Your previous spec failed the program's checks\n\n" + "\n".join(f"- {e}" for e in errs)
              + "\n\nFix every problem (change records, facts, rules or texts as needed) and output the complete corrected "
                "JSON object only.\n\nPrevious spec:\n" + prev)
    return p


def call(model, text):
    t0 = time.time()
    cmd = ["claude", "-p", "--model", model, "--tools", "", "--setting-sources", "", "--no-session-persistence",
           "--output-format", "json", "--system-prompt", SYS]
    r = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=1800, cwd="/tmp")
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None, {"error": f"cli output not json (rc {r.returncode}): {r.stdout[-300:]} {r.stderr[-300:]}", "sec": round(time.time() - t0, 1)}
    mu = next(iter((d.get("modelUsage") or {}).values()), {})
    meta = {"sec": round(time.time() - t0, 1), "is_error": d.get("is_error"), "notional_usd": d.get("total_cost_usd"),
            "model_id": next(iter(d.get("modelUsage") or {}), None), "in": mu.get("inputTokens"), "out": mu.get("outputTokens"),
            "thinking": mu.get("thinkingTokens"), "cache_read": mu.get("cacheReadInputTokens"),
            "cache_write": mu.get("cacheCreationInputTokens")}
    return d.get("result"), meta


def parse(txt):
    if not txt:
        raise ValueError("empty answer")
    s, e = txt.find("{"), txt.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("no JSON object in the answer")
    return json.loads(txt[s:e + 1])


def one(model, pid):
    b = BRIEFS[pid]
    out = OUT / model / f"{pid}.json"
    logf = OUT / model / f"{pid}.log.jsonl"
    if out.exists():
        return pid, "exists"
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = errs = None
    best = None
    for rnd in range(MAX_REPAIRS + 1):
        txt, meta = call(model, prompt(b, prev, errs))
        rec = {"round": rnd, **meta}
        try:
            spec = parse(txt)
            raw = json.dumps(spec, indent=1)
            rep, errs = engine.validate(json.loads(raw), b)
        except (ValueError, json.JSONDecodeError) as e:
            spec, raw, rep, errs = None, txt or "", None, [f"the answer is not a valid JSON object: {e}"]
        n_ok = sum(c["ok"] for c in rep["cases"].values()) if rep else 0
        rec.update(n_errors=len(errs), errors=errs[:60], proc_ok=rep is not None, n_cases_ok=n_ok)
        with open(logf, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rep is not None and (best is None or n_ok >= best[2]):
            best = (spec, rep, n_ok, rnd)
        if not errs:
            break
        prev = raw[:200000]
    if best is None or best[2] < MIN_CASES:
        json.dump({"procedure_id": pid, "rejected": True, "rounds": rnd + 1,
                   "best_cases_ok": best[2] if best else 0}, open(out, "w"), indent=1)
        return pid, f"rejected after {rnd + 1} rounds"
    spec, rep, n_ok, r_best = best
    spec["_check"] = {"round_accepted": r_best, "rounds_used": rnd + 1, "cases_ok": n_ok,
                      "cases_dropped": sorted(c for c, x in rep["cases"].items() if not x["ok"])}
    json.dump(spec, open(out, "w"), indent=1)
    return pid, f"ok round {r_best}, {n_ok}/8 cases"


if __name__ == "__main__":
    args = sys.argv[1:]
    workers = 3
    if "--workers" in args:
        i = args.index("--workers"); workers = int(args[i + 1]); args = args[:i] + args[i + 2:]
    model, pids = args[0], (args[1:] or list(BRIEFS))
    lock = threading.Lock()
    with ThreadPoolExecutor(workers) as ex:
        for pid, st in ex.map(lambda p: one(model, p), pids):
            with lock:
                print(model, pid, st, flush=True)
