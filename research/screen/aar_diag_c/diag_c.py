"""E4-C: the diagnostic test of ../aar_diag (A5an branch) with published or fixed SOPBench labels and a Haiku or Sonnet
predictor. python3 diag_c.py run|score|count published|fixed haiku|sonnet [workers]
Changes from the original: label source, 24 broken cases dropped under fixed labels (same split otherwise), output
file names, model. Prompts unchanged."""
import csv, json, re, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; ROOT = HERE.resolve().parents[2]; D = ROOT / "data" / "sopbench"
LABELS, MODEL = sys.argv[2], sys.argv[3]          # published|fixed, haiku|sonnet
FIXD = ROOT / "data" / "sopbench_fixed"
KEEP = set(json.load(open(FIXD / "cases.json"))) if LABELS == "fixed" else None   # SOPBench-fixed drops 24 broken cases
AG = ["gpt-5-mode_fc", "gemini-2.5-flash-mode_fc", "claude-3-7-sonnet-20250219-mode_fc", "gpt-4o-mini-mode_fc"]
NAME = ["GPT-5", "Gemini 2.5 Flash", "Claude 3.7 Sonnet", "GPT-4o mini"]
rows = list(csv.DictReader(open(D / "tasks.csv")))
text = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(D / "statements.jsonl")}
resp = {json.loads(l)["subject_id"]: json.loads(l)["responses"] for l in open(D / "responses.jsonl")}
if LABELS == "fixed": resp = json.load(open(FIXD / "published_agents_responses_orig_and_fixed.json"))["fixed"]
rng = np.random.default_rng(0)
byp = {}
for r in rows: byp.setdefault(r["procedure_id"], []).append(r["case_id"])
hist, test = {}, {}
for p, c in sorted(byp.items()):
    if len(c) < 2: continue
    c = list(rng.permutation(c)); h = len(c) // 2
    hist[p], test[p] = c[:h], c[h:h + 6]
if KEEP is not None:
    hist = {p: [c for c in v if c in KEEP] for p, v in hist.items()}
    test = {p: [c for c in v if c in KEEP] for p, v in test.items()}
    test = {p: v for p, v in test.items() if v}
dom = lambda p: p.split("/")[0]
cnt = lambda a, cs: (sum(resp[a][c] for c in cs), len(cs))
base = {a: np.mean([resp[a][c] for p in hist for c in hist[p]]) for a in AG}

def stats_block(p, own):
    L = ["Past results (successes / cases) of each agent on procedures of this domain:",
         "procedure | " + " | ".join(NAME)]
    for q in sorted(hist):
        if dom(q) != dom(p) or (q == p and not own): continue
        L.append(f"{q.split('/')[1]}{' (THIS procedure)' if q == p else ''} | " +
                 " | ".join("%d/%d" % cnt(a, hist[q]) for a in AG))
    return "\n".join(L)

def prompt(p, cs, v):
    P = ["AI agents handle customer-service cases with tools and a policy. A case counts as a success only if the "
         "agent's final actions and database state are exactly right (it performs the action when the policy allows "
         "it and refuses when it does not).",
         "For every case below and every agent, estimate the probability (0-100) that the agent succeeds.",
         "Agents: " + ", ".join(f"A{i+1} = {n}" for i, n in enumerate(NAME)) + "."]
    if v >= 1:
        P.append(f"Category: domain = {dom(p)}, procedure = {p.split('/')[1]}. Other procedures of this domain: " +
                 ", ".join(q.split('/')[1] for q in sorted(byp) if dom(q) == dom(p) and q != p) + ".")
    if v >= 2: P.append(stats_block(p, own=(v == 3)))
    for i, c in enumerate(cs): P.append(f"=== Case C{i+1} ===\n{text[c]}")
    P.append('Answer with JSON only, no other text: {"C1": {"A1": 0-100, "A2": ..., "A3": ..., "A4": ...}, "C2": {...}}')
    return "\n\n".join(P)

jobs = [(p, test[p][i:i + 5], v) for p in sorted(test) for i in range(0, len(test[p]), 5) for v in range(4)]
OUT = HERE / f"preds_{MODEL}_{LABELS}.jsonl"; lock = threading.Lock()

def call(job):
    p, cs, v = job
    for _ in range(3):
        r = subprocess.run(["claude", "-p", "--model", MODEL, "--tools", "", "--setting-sources", "",
                            "--no-session-persistence"], input=prompt(p, cs, v), capture_output=True, text=True,
                           timeout=600, cwd="/tmp")
        m = re.search(r"\{.*\}", r.stdout, re.S)
        try:
            d = json.loads(m.group(0))
            pr = {c: [float(d[f"C{i+1}"][f"A{j+1}"]) / 100 for j in range(4)] for i, c in enumerate(cs)}
            with lock: open(OUT, "a").write(json.dumps({"v": v, "p": p, "pred": pr}) + "\n")
            return
        except Exception: pass
    print("FAILED", p, v, r.stdout[-200:], r.stderr[-200:], flush=True)

def auc(y, s):
    y, s = np.array(y), np.array(s); a, b = s[y == 1], s[y == 0]
    if len(a) == 0 or len(b) == 0: return np.nan
    return ((a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean())

if sys.argv[1] == "count":
    print(len(test), "procedures,", sum(map(len, test.values())), "test cases,", len(jobs), "calls; prompt chars V3 ex:",
          len(prompt(jobs[3][0], jobs[3][1], 3)))
    print(prompt("bank/apply_credit_card", test["bank/apply_credit_card"][:1], 3)[:1800])
elif sys.argv[1] == "run":
    done = {(json.loads(l)["v"], c) for l in open(OUT) for c in json.loads(l)["pred"]} if OUT.exists() else set()
    todo = [j for j in jobs if (j[2], j[1][0]) not in done]
    import os
    if os.environ.get("VONLY"): todo = [j for j in todo if j[2] in {int(x) for x in os.environ["VONLY"].split(",")}]
    print("todo", len(todo), flush=True)
    with ThreadPoolExecutor(int(sys.argv[4]) if len(sys.argv) > 4 else 4) as ex: list(ex.map(call, todo))
else:
    P = {}
    for l in open(OUT):
        d = json.loads(l)
        for c, pr in d["pred"].items(): P[(d["v"], c)] = pr
    T = [(p, c) for p in sorted(test) for c in test[p] if all((v, c) in P for v in range(4))]
    print(f"test cases scored {len(T)} of {sum(map(len, test.values()))}, procedures {len({p for p, _ in T})}")
    Y = np.array([[resp[a][c] for a in AG] for _, c in T]); pid = np.array([p for p, _ in T])
    def proc_rate(p, a, own=True):
        k, n = cnt(a, hist[p]); return (k + 2 * base[a]) / (n + 2)
    def dom_rate(p, a):
        cs = [c for q in hist if dom(q) == dom(p) and q != p for c in hist[q]]
        return np.mean([resp[a][c] for c in cs]) if cs else base[a]
    M = {"BaseRate": np.array([[base[a] for a in AG] for _ in T]),
         "V0 Vanilla (text only)": np.array([P[(0, c)] for _, c in T]),
         "V1 +Category": np.array([P[(1, c)] for _, c in T]),
         "DomainRate (no LLM)": np.array([[dom_rate(p, a) for a in AG] for p, _ in T]),
         "V2 +Stats similar procedures": np.array([P[(2, c)] for _, c in T]),
         "ProcRate (no LLM)": np.array([[proc_rate(p, a) for a in AG] for p, _ in T]),
         "V3 +Stats this procedure": np.array([P[(3, c)] for _, c in T]),
         "Oracle": Y.astype(float)}
    def metrics(idx):
        o = {}
        for k, S in M.items():
            y, s = Y[idx], S[idx]; b0 = ((M["BaseRate"][idx] - y) ** 2).mean(0)
            top = [y[np.argsort(-(s[:, j] + 1e-9 * np.arange(len(s))))[:len(s) // 2], j].mean() for j in range(4)]
            o[k] = (np.nanmean([auc(y[:, j], s[:, j]) for j in range(4)]), (1 - ((s - y) ** 2).mean(0) / b0).mean(), np.mean(top))
        return o
    full = metrics(np.arange(len(T))); up = sorted(set(pid)); B = []
    for _ in range(1000):
        idx = np.concatenate([np.where(pid == q)[0] for q in rng.choice(up, len(up))]); B.append(metrics(idx))
    print(f"success rate of the four agents on test: {Y.mean():.3f}")
    print(f"{'method':32s} {'AUC [95% CI]':>24s} {'Brier skill [95% CI]':>26s} {'success in top half':>22s}")
    for k in M:
        ci = lambda j: (np.nanpercentile([b[k][j] for b in B], 2.5), np.nanpercentile([b[k][j] for b in B], 97.5))
        print(f"{k:32s} {full[k][0]:6.3f} [{ci(0)[0]:.3f}, {ci(0)[1]:.3f}] {full[k][1]:+8.3f} [{ci(1)[0]:+.3f}, {ci(1)[1]:+.3f}] "
              f"{full[k][2]:8.3f} [{ci(2)[0]:.3f}, {ci(2)[1]:.3f}]")
    for a, b in [("V1 +Category", "V0 Vanilla (text only)"), ("V2 +Stats similar procedures", "V1 +Category"),
                 ("V3 +Stats this procedure", "V2 +Stats similar procedures"), ("V3 +Stats this procedure", "ProcRate (no LLM)"),
                 ("V2 +Stats similar procedures", "DomainRate (no LLM)")]:
        d = [x[a][1] - x[b][1] for x in B]
        print(f"Brier skill {a} minus {b}: {full[a][1]-full[b][1]:+.3f} [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]")
    json.dump({k: list(map(float, v)) for k, v in full.items()}, open(HERE / f"results_{MODEL}_{LABELS}.json", "w"), indent=1)
