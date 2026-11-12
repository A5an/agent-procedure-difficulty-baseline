"""TauIndianBank episodes with Claude agents through Claude Code headless (subscription), each episode a fresh session
with no other context, in its own working directory, talking to the bank only through bridge.py. The customer is a
Claude model as well (default haiku) with tau2's user-simulator prompt. Scoring is tau2's own evaluate_simulation
(ACTION + DB, as for the Gemini runs).

  <tau venv python> run_claude_tib.py --agent sonnet|haiku --set real|k0|k1|k2|k5|foreign [--sim haiku] [--workers 6]
                                      [--families f1 f2 ...]   (default: all 50; the seedk pilot five with --pilot)
Output: runs/<set>_<agent>.jsonl (one line per episode, resumable), sessions/<set>_<agent>/<task_id>/ (trajectory)."""
import argparse, json, os, shutil, subprocess, sys, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent; SCR = HERE.parent
sys.path.insert(0, str(SCR / "tib")); sys.path.insert(0, str(HERE))
from common import load_tasks, family, DOM
import bridge as BR
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.domains.indian_banking.environment import get_environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

TAU_PY = sys.executable
ap = argparse.ArgumentParser()
ap.add_argument("--agent", required=True); ap.add_argument("--set", required=True); ap.add_argument("--sim", default="haiku")
ap.add_argument("--workers", type=int, default=6); ap.add_argument("--families", nargs="*"); ap.add_argument("--pilot", action="store_true")
ap.add_argument("--limit", type=int)
A = ap.parse_args()
SEL = json.load(open(SCR / "seedk" / "selection_seedk.json")); FAMS = set(A.families or (SEL["pilot_families"] if A.pilot else SEL["families"]))


def tasks_for(which):
    if which == "real":
        allt = {t["id"]: t for t in load_tasks()}
        return [allt[i] for f in sorted(SEL["families"]) for i in SEL["families"][f]["eval_tasks"]]
    if which == "k0": return json.load(open(SCR / "tib" / "probes_v2.json")) + json.load(open(SCR / "combo_rep" / "probes_doc.json"))
    if which == "k2": return json.load(open(SCR / "tib" / "seeded" / "probes.json")) + json.load(open(SCR / "combo_rep" / "probes_seeded.json"))
    d = SCR / "seedk" / "probes" / which
    return [t for p in sorted(d.glob("*.json")) for t in json.load(open(p))]


TASKS = [t for t in tasks_for(A.set) if family(t) in FAMS]
if A.limit: TASKS = TASKS[:A.limit]
name = f"{A.set}_{A.agent}" + ("" if A.sim == "haiku" else f"_sim{A.sim}")
OUT = HERE / "runs" / f"{name}.jsonl"; SESS = HERE / "sessions" / name; WORK = HERE / "work" / name
OUT.parent.mkdir(exist_ok=True); SESS.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
done = set()
if OUT.exists():
    for l in open(OUT):
        r = json.loads(l)
        if r.get("reward") is not None: done.add(r["task_id"])
todo = [t for t in TASKS if t["id"] not in done]
print(f"{name}: {len(TASKS)} tasks, {len(done)} done, {len(todo)} to run", flush=True)

INSTR = (DOM / "agent_instruction.txt").read_text().strip(); POLICY = (DOM / "policy.md").read_text().strip()
TOOLS = [t.openai_schema["function"] for t in get_environment().get_tools()]
GREETING = "Hi! How can I help you today?"
MAX_ROUNDS = 30   # agent turns (messages to the customer); tau2 allows 200 steps
SYSTEM = f"""<instructions>
{INSTR}
</instructions>
<policy>
{POLICY}
</policy>

<how_you_act>
You are the bank's assistant in a live chat with a customer. Bank tools run through one command-line program:
  python3 {{bank}} call <tool_name> '<json arguments>'
Run it exactly in this form from the current directory, one tool call per Bash command, without cd, pipes or
&&; its output is the tool's result. To talk to the customer, end your turn: the text of
your final reply is sent to the customer as your chat message, and the customer's answer comes back as your next
prompt, starting with "CUSTOMER:". So every turn ends with a message written to the customer, never with notes to
yourself. Keep going until the customer's request is handled; the customer ends the chat when done.
</how_you_act>"""
LIMIT = ("usage limit", "rate limit", "rate_limit", "overloaded", "429", "limit reached", "too many requests", "resets")
lock = threading.Lock(); stop = {"v": False, "last_ok": time.time()}


def episode(t):
    if stop["v"]: return
    sd = SESS / t["id"]; wd = WORK / t["id"]; t0 = time.time(); attempt = 0
    while True:
        shutil.rmtree(sd, ignore_errors=True); shutil.rmtree(wd, ignore_errors=True); sd.mkdir(parents=True); wd.mkdir(parents=True)
        meta = {"task": t, "sim_model": A.sim, "workdir": str(wd), "session": str(sd)}
        json.dump(meta, open(sd / "meta.json", "w"))
        bank = wd / "bank.py"   # in the agent's working directory, so the plain relative command works
        bank.write_text(f"import os, sys\nos.execv({TAU_PY!r}, [{TAU_PY!r}, {str(HERE / 'bridge.py')!r}, {str(sd)!r}] + sys.argv[1:])\n")
        start = datetime.now().isoformat(); term = None
        try:
            g = AssistantMessage(role="assistant", content=GREETING); BR.append(sd, g)
            first = BR.user_reply(meta, [g]); BR.append(sd, UserMessage(role="user", content=first))
        except Exception as e:
            res = {"is_error": True, "result": f"simulator: {e}"}; break
        sid = str(uuid.uuid4())
        prompt = ("Bank tools (OpenAI function schemas):\n" + json.dumps(TOOLS, indent=0) + f"\n\nThe chat has started. You said: \"{GREETING}\"\n\n"
                  f"CUSTOMER: {first}")
        base = ["--model", A.agent, "--system-prompt", SYSTEM.replace("{bank}", "bank.py"),
                "--permission-mode", "dontAsk", "--allowedTools", "Bash(python3 bank.py:*)", f"Bash(python3 {bank}:*)", f"Bash(python3 ./bank.py:*)",
                "--disallowedTools", "Read Edit Write Glob Grep WebFetch WebSearch Agent NotebookEdit Skill TodoWrite",
                "--setting-sources", "", "--strict-mcp-config", "--output-format", "json"]
        rounds, turns, cost, res, term = 0, 0, 0.0, {}, None
        nxt = ["claude", "-p", prompt, "--session-id", sid] + base
        restart = False
        while rounds < MAX_ROUNDS:
            p = None
            try:
                p = subprocess.run(nxt, cwd=str(wd), capture_output=True, text=True, timeout=900); res = json.loads(p.stdout)
            except subprocess.TimeoutExpired: res = {"is_error": True, "result": "timeout"}
            except Exception: res = {"is_error": True, "result": "unparsable: " + ((p.stdout[:300] + p.stderr[:300]) if p else "")}
            txt = (str(res.get("result"))[:600] + str(res.get("api_error_status")) + (p.stderr[:300] if p else "")).lower()
            if res.get("is_error") and any(w in txt for w in LIMIT):
                attempt += 1
                with lock:
                    if time.time() - stop["last_ok"] > 1800: stop["v"] = True
                if stop["v"]: print("STOP: limit errors for 30 minutes", flush=True); return
                print("limit-like error, backoff", t["id"], txt[:120], flush=True); time.sleep(min(120 * attempt, 600))
                if rounds == 0: restart = True; break
                continue
            rounds += 1; turns += res.get("num_turns") or 0; cost += res.get("total_cost_usd") or 0.0
            if res.get("is_error"): break
            _, msgs = BR.load(sd)
            if BR.ended(msgs): term = TerminationReason.USER_STOP; break
            text = (res.get("result") or "").strip()
            if not text: term = TerminationReason.AGENT_STOP; break
            am = AssistantMessage(role="assistant", content=text); BR.append(sd, am)
            try: reply = BR.user_reply(meta, msgs + [am])
            except Exception as e: res = {"is_error": True, "result": f"simulator: {e}"}; break
            BR.append(sd, UserMessage(role="user", content=reply))
            _, msgs = BR.load(sd)
            if BR.ended(msgs): term = TerminationReason.USER_STOP; break
            nxt = ["claude", "-p", "CUSTOMER: " + reply, "--resume", sid] + base
        if restart: continue
        if term is None and rounds >= MAX_ROUNDS: term = TerminationReason.MAX_STEPS
        res["num_turns"], res["total_cost_usd"], res["rounds"] = turns, cost, rounds
        break
    _, msgs = BR.load(sd)
    rec = {"task_id": t["id"], "agent": A.agent, "set": A.set, "sim": A.sim, "sec": round(time.time() - t0, 1),
           "agent_turns": res.get("num_turns"), "agent_cost_equiv": res.get("total_cost_usd"), "n_msgs": len(msgs)}
    rec["rounds"] = res.get("rounds")
    tr = Path.home() / ".claude" / "projects" / str(wd).replace("/", "-").replace("_", "-")
    rec["denied"] = sum(open(f).read().count("has been denied") for f in tr.glob("*.jsonl")) if tr.exists() else None
    if res.get("is_error") and res.get("result") != "timeout" and term is None:
        rec.update(reward=None, error=str(res.get("result"))[:300])
    else:
        term = term or (TerminationReason.USER_STOP if BR.ended(msgs) else TerminationReason.AGENT_STOP)
        task = Task.model_validate(t)
        sim = SimulationRun(id=str(uuid.uuid4()), task_id=t["id"], start_time=start, end_time=datetime.now().isoformat(),
                            duration=time.time() - t0, termination_reason=term, messages=msgs)
        try:
            ri = evaluate_simulation(sim, task, EvaluationType.ALL, solo_mode=False, domain="indian_banking")
            rec.update(reward=float(ri.reward), termination=str(term), db_ok=bool(ri.db_check.db_match) if ri.db_check else None,
                       actions_ok=[bool(x.action_match) for x in (ri.action_checks or [])], timeout=res.get("result") == "timeout")
        except Exception as e:
            rec.update(reward=None, error=f"evaluation: {type(e).__name__}: {str(e)[:200]}")
    with lock:
        with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
        if rec.get("reward") is not None: stop["last_ok"] = time.time()
    print(t["id"], rec.get("reward"), rec.get("termination", rec.get("error")), rec["sec"], rec.get("agent_turns"), flush=True)


with ThreadPoolExecutor(A.workers) as ex:
    list(ex.map(episode, todo))
print("done; stopped_early =", stop["v"], flush=True)
