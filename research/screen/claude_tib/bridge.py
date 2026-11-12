"""Bridge between a Claude Code agent session and the TauIndianBank environment (tau venv). One session directory per
episode holds meta.json (task, simulator model) and msgs.jsonl (the tau2 message trajectory so far).

  bridge.py <session_dir> call <tool_name> '<json arguments>'   run one bank tool, print its output
  bridge.py <session_dir> say <message to the customer>         send a message, print the customer's reply

The environment state is rebuilt for every call by tau2's own replay (Environment.set_state over the trajectory), the
same mechanism the evaluator uses. The customer is played by a Claude model through `claude -p` with tau2's user
simulator system prompt; it sees only the text messages, as in tau2."""
import json, os, subprocess, sys, uuid
from pathlib import Path
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.tasks import Task
from tau2.domains.indian_banking.environment import get_environment
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import OUT_OF_SCOPE, STOP, TRANSFER

ROLE = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage}
ENDED = "[The customer has ended the conversation. Do not send more messages; stop now.]"


def load(sd):
    meta = json.load(open(sd / "meta.json"))
    msgs = [ROLE[d["role"]].model_validate(d) for d in map(json.loads, open(sd / "msgs.jsonl"))] if (sd / "msgs.jsonl").exists() else []
    return meta, msgs


def append(sd, *ms):
    with open(sd / "msgs.jsonl", "a") as f:
        for m in ms: f.write(json.dumps(m.model_dump(mode="json")) + "\n")


def ended(msgs):
    return any(isinstance(m, UserMessage) and m.content and any(t in m.content for t in (STOP, TRANSFER, OUT_OF_SCOPE)) for m in msgs)


def user_reply(meta, msgs):
    task = Task.model_validate(meta["task"])
    system = UserSimulator(llm="claude", instructions=str(task.user_scenario)).system_prompt
    lines = []
    for m in msgs:
        if isinstance(m, AssistantMessage) and m.content: lines.append(f"ASSISTANT: {m.content}")
        elif isinstance(m, UserMessage) and m.content: lines.append(f"YOU: {m.content}")
    prompt = ("This is the conversation so far between you (the customer) and the bank's virtual assistant:\n\n" + "\n\n".join(lines)
              + "\n\nWrite only your next message to the assistant, as the customer, following the guidelines and your scenario. "
                "If the conversation should end, end it the way the guidelines say.")
    for attempt in range(4):
        try:
            p = subprocess.run(["claude", "-p", prompt, "--model", meta["sim_model"], "--system-prompt", system, "--tools", "",
                                "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence", "--output-format", "json"],
                               capture_output=True, text=True, timeout=300, cwd=str(meta["workdir"]))
        except subprocess.TimeoutExpired:   # added 12 Nov after a stalled call crashed one runner; retried like other failures
            import time; time.sleep(30 * (attempt + 1)); continue
        try:
            r = json.loads(p.stdout)
            if not r.get("is_error") and (r.get("result") or "").strip():
                with open(Path(meta["session"]) / "sim_usage.jsonl", "a") as f: f.write(json.dumps({"usage": r.get("usage"), "cost": r.get("total_cost_usd")}) + "\n")
                return r["result"].strip()
        except Exception:
            pass
    raise RuntimeError("user simulator failed after 4 attempts")


def main():
    sd = Path(sys.argv[1]); cmd = sys.argv[2] if len(sys.argv) > 2 else ""
    meta, msgs = load(sd)
    if ended(msgs):
        print(ENDED); return
    if cmd == "call":
        if len(sys.argv) < 4: print("usage: call <tool_name> '<json arguments>'"); return
        name = sys.argv[3]; raw = sys.argv[4] if len(sys.argv) > 4 else "{}"
        try: args = json.loads(raw) if raw.strip() else {}
        except Exception as e: print(f"Error: arguments are not valid JSON ({e})"); return
        if not isinstance(args, dict): print("Error: arguments must be a JSON object"); return
        task = Task.model_validate(meta["task"])
        env = get_environment()
        env.set_state(task.initial_state.initialization_data, task.initial_state.initialization_actions, msgs)
        tc = ToolCall(id="c" + uuid.uuid4().hex[:10], name=name, arguments=args, requestor="assistant")
        tm = env.get_response(tc)
        append(sd, AssistantMessage(role="assistant", content=None, tool_calls=[tc]), tm)
        print(tm.content)
    elif cmd == "say":
        text = " ".join(sys.argv[3:]).strip()
        if not text: print("usage: say <message to the customer>"); return
        am = AssistantMessage(role="assistant", content=text); append(sd, am)
        reply = user_reply(meta, msgs + [am])
        append(sd, UserMessage(role="user", content=reply))
        print("CUSTOMER: " + reply)
        if ended(msgs + [am, UserMessage(role="user", content=reply)]): print(ENDED)
    else:
        print("commands: call <tool_name> '<json arguments>' | say <message to the customer>")


if __name__ == "__main__":
    main()
