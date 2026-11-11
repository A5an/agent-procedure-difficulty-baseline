"""Exp 2: features of every recorded agent episode at its commit point (no API calls).

Commit point = the first call to a state-changing tool (SOPBench: the case's target action; tau2 airline/retail and
TauIndianBank: their write tools). The features use only what happened before that call, plus the call's own arguments.
Episodes with no write are cut at their end (the agent's last message). Label = episode failed.

Sources: SOPBench release logs (function-calling configs only), tau2-bench airline + retail submissions (15 x 4 trials),
our TauIndianBank runs (real tasks, both agents). Output: episodes.csv.
"""
import ast, glob, json, os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "..", "..", "data", "external")
TAU_WRITES = {"airline": {"book_reservation", "cancel_reservation", "update_reservation_baggages", "update_reservation_flights",
                          "update_reservation_passengers", "send_certificate"},
              "retail": {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
                         "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
                         "return_delivered_order_items"}}
TIB_WRITES = {"create_fd", "create_rd", "close_deposit", "update_deposit_renewal", "toggle_card_freeze", "block_card",
              "set_card_controls", "cancel_mandate", "stop_cheque_payment", "update_address", "raise_request"}
CONFIRM = re.compile(r"\b(yes|yeah|yep|confirm|confirmed|proceed|go ahead|sure|please do|correct|that's right|ok|okay|do it)\b", re.I)


def leaves(x):
    if isinstance(x, dict):
        for v in x.values(): yield from leaves(v)
    elif isinstance(x, list):
        for v in x: yield from leaves(v)
    elif x is not None:
        yield str(x)


def features(events, writes):
    """events: list of dicts kind in {user, say, call, tool}; returns feature dict at the commit point"""
    seen_text, calls, n_err, n_empty, n_rep, n_user, n_say, chars = [], [], 0, 0, 0, 0, 0, 0
    last_user, last_say, commit = "", "", None
    keys = set()
    for e in events:
        if e["kind"] == "call":
            if e["name"] in writes:
                commit = e; break
            k = (e["name"], json.dumps(e["args"], sort_keys=True, default=str))
            n_rep += k in keys; keys.add(k); calls.append(e["name"])
            if e["args"] and all(v in ([], {}, "", None) for v in e["args"].values()): n_empty += 1
        elif e["kind"] == "tool":
            seen_text.append(e["text"]); chars += len(e["text"])
            t = e["text"].lower()
            n_err += bool(e.get("error")) or t.startswith("error") or '"error_code"' in t
        elif e["kind"] == "user":
            seen_text.append(e["text"]); last_user = e["text"]; n_user += 1; chars += len(e["text"])
        elif e["kind"] == "say":
            last_say = e["text"]; n_say += 1; chars += len(e["text"])
    f = {"has_commit": int(commit is not None), "n_calls": len(calls), "n_distinct_tools": len(set(calls)),
         "n_tool_errors": n_err, "n_empty_arg_calls": n_empty, "n_repeat_calls": n_rep, "n_user_msgs": n_user,
         "n_agent_texts": n_say, "log_chars": np.log1p(chars),
         "user_confirmed": int(bool(CONFIRM.search(last_user))) if commit else 0,
         "agent_asked": int("?" in last_say or "confirm" in last_say.lower()) if commit else 0}
    if commit:
        vals = [v for v in leaves(commit["args"]) if len(v) >= 3]
        blob = "\n".join(seen_text).lower()
        f["n_commit_args"] = len(vals)
        f["grounding"] = float(np.mean([v.lower() in blob for v in vals])) if vals else 1.0
    else:
        f["n_commit_args"] = 0; f["grounding"] = 1.0
    return f


def sop():
    rows = []
    for fn in glob.glob(os.path.join(EXT, "SOPBench-d2622008", "output", "*", "ast_*-mode_fc-*.json")):
        dom = os.path.basename(os.path.dirname(fn)); agent = re.search(r"ast_(.*)-dep_", os.path.basename(fn)).group(1)
        cnt = {}
        for r in json.load(open(fn)):
            g = r["task"]["user_goal"]; cid = f"{dom}/{g}/{cnt.get(g, 0)}"; cnt[g] = cnt.get(g, 0) + 1
            if not r.get("evaluations") or not r.get("interactions"): continue
            ev = []
            for m in r["interactions"][0]["interaction"]:
                if m.get("sender") == "user": ev.append({"kind": "user", "text": str(m.get("content") or "")})
                elif "tool_name" in m: ev.append({"kind": "tool", "text": str(m.get("content")), "error": False})
                else:
                    tcs = m.get("tool_calls") or []
                    if isinstance(tcs, str):
                        try: tcs = ast.literal_eval(tcs)
                        except Exception: tcs = []
                    if m.get("content"): ev.append({"kind": "say", "text": str(m["content"])})
                    for tc in tcs:
                        fnc = tc.get("function", tc); a = fnc.get("arguments") or "{}"
                        try: a = json.loads(a) if isinstance(a, str) else a
                        except Exception: a = {}
                        ev.append({"kind": "call", "name": fnc.get("name"), "args": a if isinstance(a, dict) else {}})
            f = features(ev, {g})
            rows.append({"bench": "sopbench", "agent": agent, "task": cid, "proc": f"{dom}/{g}",
                         "fail": 1 - int(bool(r["evaluations"][0]["success"])), **f})
    return rows


def tau_events(msgs):
    ev = []
    for m in msgs:
        if m["role"] == "user": ev.append({"kind": "user", "text": str(m.get("content") or "")})
        elif m["role"] == "tool": ev.append({"kind": "tool", "text": str(m.get("content") or ""), "error": m.get("error") in (True, "True")})
        elif m["role"] == "assistant":
            if m.get("content"): ev.append({"kind": "say", "text": str(m["content"])})
            for tc in m.get("tool_calls") or []:
                ev.append({"kind": "call", "name": tc.get("name"), "args": tc.get("arguments") or {}})
    return ev


def tau2():
    rows = []
    for dom in ("airline", "retail"):
        for fn in glob.glob(os.path.join(EXT, "tau2-domains", "trajectories", f"*{dom}*.json")):
            agent = os.path.basename(fn).split("__")[0]
            d = json.load(open(fn))
            for s in d["simulations"]:
                ri = s.get("reward_info") or {}
                if ri.get("reward") is None: continue
                f = features(tau_events(s["messages"]), TAU_WRITES[dom])
                rows.append({"bench": f"tau2_{dom}", "agent": agent, "task": f"{dom}/{s['task_id']}", "proc": f"{dom}/{s['task_id']}",
                             "fail": 1 - int(float(ri["reward"]) >= 1.0), **f})
    return rows


def tib():
    rows = []
    T = os.path.join(HERE, "..", "tib")
    sel = json.load(open(os.path.join(T, "selection.json")))
    for a in ("g38", "flash"):
        for l in open(os.path.join(T, "runs", f"real_{a}.jsonl")):
            r = json.loads(l)
            msgs = json.load(open(os.path.join(T, "runs", "msgs", f"real_{a}", f"{r['task_id']}.json")))
            fam = next(f for f in sel["test_families"] if f"_{f}_" in r["task_id"])
            f = features(tau_events(msgs), TIB_WRITES)
            rows.append({"bench": "tib", "agent": a, "task": r["task_id"], "proc": fam, "fail": 1 - int(r["reward"] >= 1), **f})
    return rows


if __name__ == "__main__":
    df = pd.DataFrame(sop() + tau2() + tib())
    df.to_csv(os.path.join(HERE, "episodes.csv"), index=False)
    print(df.groupby("bench").agg(n=("fail", "size"), fail=("fail", "mean"), agents=("agent", "nunique"), commit=("has_commit", "mean")))
