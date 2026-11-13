"""Pilot bookkeeping for E1 and E2 (no rho on the pilot, PREREG.md): valid probes and attempts per condition and family,
drop reasons, episodes finished and errors, USD by phase. Phases in the ledger: tag gen = E1 generation; agent/user calls
before the timestamp in phase_marks.json = E2 pilot episodes, after it = E1 pilot episodes."""
import json, collections, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
L = [json.loads(l) for l in open(f"{HERE}/ledger.jsonl")]
marks = json.load(open(f"{HERE}/phase_marks.json")) if os.path.exists(f"{HERE}/phase_marks.json") else {}
t_e1 = marks.get("e1_episodes_start", 1e18)
usd = collections.Counter()
for x in L:
    ph = "E1 generation" if x.get("tag") == "gen" else ("E2 episodes" if x.get("t", 0) < t_e1 else "E1 episodes")
    usd[ph] += x.get("usd", 0)
print("USD by phase:", {k: round(v, 2) for k, v in usd.items()}, "total", round(sum(usd.values()), 2))
log = [json.loads(l) for l in open(f"{HERE}/probes_log.jsonl")] if os.path.exists(f"{HERE}/probes_log.jsonl") else []
by = collections.defaultdict(lambda: [0, 0]); drops = collections.Counter()
for r in log:
    by[(r["cond"], r["family"])][1] += 1
    if r.get("kept"): by[(r["cond"], r["family"])][0] += 1
    else: drops[(r["cond"], (r.get("drop") or "")[:40])] += 1
for k in sorted(by): print("  ", k, "kept/attempts", by[k])
print("drop reasons:", dict(drops))
gen_usd = collections.Counter()
for runs in sorted(glob.glob(f"{HERE}/runs/*.jsonl") + glob.glob(f"{HERE}/../sim2/runs/*.jsonl")):
    R = [json.loads(l) for l in open(runs)]
    ok = [r for r in R if r.get("reward") is not None]
    print(os.path.basename(runs), "episodes", len(ok), "errors", len(R) - len(ok), "mean reward", round(sum(r["reward"] for r in ok) / max(1, len(ok)), 3),
          "terminations", dict(collections.Counter(r.get("termination", "err").split(".")[-1] for r in R)))
