"""Score all finished sessions with the SOPBench evaluator through the replay path; write real_results.csv, synth_results.csv, summary."""
import os, sys, json, csv, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score as S
T = S.T
def run(kind):
    key = json.load(open(f"{HERE}/keys_{kind}.json")); rows = []
    for cid, k in key.items():
        jp = f"{HERE}/out/{kind}/{k}.json"
        if not os.path.exists(jp): continue
        r = json.load(open(jp))
        if r.get("is_error"): continue
        sd = f"{HERE}/sessions/{kind}/{k}"
        t = T._load_case(cid, kind)
        log = [e for e in T.read_log(sd) if not e.get("blocked")]
        ds, allowed, contents = T.replay(t, log)
        fc = [{"tool_name": e["tool_name"], "arguments": e["arguments"], "content": S.try_eval(c)} for e, c in zip(log, contents)]
        mism = not all(c == e["content"] for e, c in zip(log, contents))
        ev = S.evaluator_function_directed_graph(t["domain"], t, [], fc, {"final_database": ds.data}, "full")
        f = S.flags(ev, bool(t["action_should_succeed"]))
        proc = f"{t['domain']}/{t['user_goal']}"
        rows.append(dict(case_id=cid, domain=t["domain"], procedure_id=proc, perform=int(f["should"]), success=int(f["success"]),
                         unsafe_action=int(f["unsafe"]), wrongful_refusal=int(f["wrongful"]), performed=int(f["performed"]), calls=len(log),
                         usd=r.get("total_cost_usd", 0), turns=r.get("num_turns"), secs=r.get("secs"), replay_mismatch=int(mism),
                         empty_final=int(not (r.get("result") or "").strip()), denials=len(r.get("permission_denials") or [])))
    return rows
def write(rows, path, kind):
    with open(path, "w", newline="") as f:
        if kind == "real":
            w = csv.writer(f); w.writerow(["case_id", "success", "perform_label", "unsafe_action", "wrongful_refusal"])
            for r in rows: w.writerow([r["case_id"], r["success"], r["perform"], r["unsafe_action"], r["wrongful_refusal"]])
        else:
            w = csv.writer(f); w.writerow(["synthetic_case_id", "procedure_id", "perform_label", "success"])
            for r in rows: w.writerow([r["case_id"], r["procedure_id"], r["perform"], r["success"]])
def summ(rows, name):
    if not rows: return
    n = len(rows); m = lambda xs: sum(xs) / max(len(xs), 1)
    P = [r for r in rows if r["perform"]]; R = [r for r in rows if not r["perform"]]
    print(f"== {name}: n={n} success={m([r['success'] for r in rows]):.3f} perform(n={len(P)})={m([r['success'] for r in P]):.3f} refuse(n={len(R)})={m([r['success'] for r in R]):.3f}")
    print(f"   unsafe on must-refuse={m([r['unsafe_action'] for r in R]):.3f} wrongful on must-perform={m([r['wrongful_refusal'] for r in P]):.3f} calls={m([r['calls'] for r in rows]):.2f}")
    print(f"   cost total=${sum(r['usd'] for r in rows):.2f} per case=${m([r['usd'] for r in rows]):.4f} replay_mismatch={sum(r['replay_mismatch'] for r in rows)} empty_final={sum(r['empty_final'] for r in rows)} denials={sum(r['denials'] for r in rows)}")
    d = collections.defaultdict(list)
    for r in rows: d[r["domain"]].append(r)
    for k, v in sorted(d.items()):
        vp = [x for x in v if x["perform"]]; vr = [x for x in v if not x["perform"]]
        print(f"   {k:15s} n={len(v):4d} succ={m([x['success'] for x in v]):.3f} perform={m([x['success'] for x in vp]):.3f}(n={len(vp)}) refuse={m([x['success'] for x in vr]):.3f}(n={len(vr)})")
if __name__ == "__main__":
    for kind in ["real", "synth"]:
        if not os.path.exists(f"{HERE}/keys_{kind}.json"): continue
        rows = run(kind); write(rows, f"{HERE}/{kind}_results.csv", kind); summ(rows, kind)
        json.dump(rows, open(f"{HERE}/{kind}_rows.json", "w"))
