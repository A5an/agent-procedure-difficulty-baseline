"""E8 log analysis (context repo task P1-4). No API calls, reads the recorded E8 episodes.

Questions:
  Q1  how many cases the second model rescued and how many it spoiled
  Q2  is the gain concentrated on cases that a predictor calls hard
  Q3  are errors more frequent where the two models disagree

Two ways to count rescues. Cross-run: pair arm vs model A alone on the same case (two separate runs, so part of the
flips is run-to-run noise; the noise level is measured on the 5 Oct repeat of the same flash and g38 configs).
At the gate: inside a pair episode we know what A proposed (the discarded call) and what B did, so on refuse cases
the counterfactual "A acts" is known to be a failure without a second run.
"""
import os, sys, json, csv, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCREEN = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SCREEN, "e8"))
import analyze_e8 as E          # noqa: E402  (loads tasks, evaluator, sample)

AN, tasks, SAMPLE, PROC = E.AN, E.tasks, E.SAMPLE, E.PROC
SHOULD = {c: bool(tasks[c]["action_should_succeed"]) for c in SAMPLE}
FILES = {"flash": "flash.jsonl", "pro": "pro.jsonl", "pro>flash": "pro_to_flash.jsonl", "flash>pro": "flash_to_pro.jsonl",
         "lite": "lite.jsonl", "g38": "g38.jsonl", "lite>g38": "lite_to_g38.jsonl"}
PAIRS = {"pro>flash": ("pro", "flash"), "flash>pro": ("flash", "pro"), "lite>g38": ("lite", "g38")}
rng = np.random.default_rng(7)


def load_raw(fname):
    out = {}
    for l in open(os.path.join(SCREEN, "e8", "episodes", fname)):
        r = json.loads(l)
        if r.get("error") or "interaction" not in r or r["case_id"] in out: continue
        out[r["case_id"]] = r
    return out


def tool_calls(msg):
    tc = msg.get("tool_calls")
    if not tc: return []
    if isinstance(tc, str):
        try: tc = json.loads(tc)
        except Exception: return []
    out = []
    for t in tc:
        f = t.get("function", t)
        a = f.get("arguments") or "{}"
        try: a = json.loads(a) if isinstance(a, str) else a
        except Exception: a = {}
        out.append((f.get("name"), a))
    return out


def boot_ci(vals_by_proc, B=5000):
    procs = list(vals_by_proc)
    est = np.mean(np.concatenate([vals_by_proc[p] for p in procs]))
    bs = []
    for _ in range(B):
        pick = rng.integers(0, len(procs), len(procs))
        bs.append(np.mean(np.concatenate([vals_by_proc[procs[j]] for j in pick])))
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def by_proc(ids, f):
    d = collections.defaultdict(list)
    for c in ids: d[PROC[c]].append(f(c))
    return {k: np.array(v, float) for k, v in d.items()}


D = {k: E.load(v) for k, v in FILES.items()}
D["flash (5 Oct)"] = AN.load_arm("gemini-2.5-flash", "H0")
D["g38 (5 Oct)"] = AN.load_arm("gemini-3.8-flash", "H0")
RAW = {k: load_raw(FILES[k]) for k in PAIRS}
res = {}

# ---------------- Q1a: cross-run rescue / spoil, with the repeat-run noise as reference ----------------
def flips(a, b, ids):
    ids = [c for c in ids if c in D[a] and c in D[b]]
    up = sum(1 for c in ids if D[a][c]["success"] and not D[b][c]["success"])
    down = sum(1 for c in ids if not D[a][c]["success"] and D[b][c]["success"])
    return {"n": len(ids), "rescued": up, "spoiled": down, "net": up - down}

q1 = {}
for m, (a, b) in PAIRS.items():
    for ref in (a, b):
        for cls, ids in (("all", SAMPLE), ("perform", [c for c in SAMPLE if SHOULD[c]]), ("refuse", [c for c in SAMPLE if not SHOULD[c]])):
            q1[f"{m} vs {ref} alone [{cls}]"] = flips(m, ref, ids)
for a, b in (("flash", "flash (5 Oct)"), ("g38", "g38 (5 Oct)")):
    for cls, ids in (("all", SAMPLE), ("perform", [c for c in SAMPLE if SHOULD[c]]), ("refuse", [c for c in SAMPLE if not SHOULD[c]])):
        q1[f"NOISE {a} 13 Oct vs 5 Oct [{cls}]"] = flips(a, b, ids)
res["q1_cross_run"] = q1

# ---------------- Q1b: decisions at the gate (within the pair episode) ----------------
gate = {}
GATE_CASE = {}
for m in PAIRS:
    rows = collections.Counter()
    for c in SAMPLE:
        if c not in D[m] or c not in RAW[m]: continue
        r = RAW[m][c]; log = (r.get("router") or {}).get("log") or []
        if not log:
            rows[("no handoff", SHOULD[c], D[m][c]["success"])] += 1; GATE_CASE[(m, c)] = "none"; continue
        prop = log[0]["a_proposal"]; first = log[0]["b_first"]
        try: pargs = json.loads(prop.get("args") or "{}")
        except Exception: pargs = {}
        # did B ever issue the target call, and with A's arguments?
        b_calls = [tc for msg in r["interaction"] if msg.get("role") == "assistant" for tc in tool_calls(msg) if tc[0] == prop["call"]]
        same_first = first.get("call") == prop["call"]
        kind = "B repeats A's call" if same_first else ("B calls it later" if b_calls else "B never calls it")
        rows[(kind, SHOULD[c], D[m][c]["success"])] += 1
        GATE_CASE[(m, c)] = "agree" if same_first else "deviate"
    tab = {}
    for (kind, should, succ), n in sorted(rows.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        tab[f"{kind} | {'perform' if should else 'refuse'} | {'success' if succ else 'fail'}"] = n
    gate[m] = tab
res["q1_gate"] = gate

# ---------------- Q2: gain vs predicted difficulty ----------------
fc = {r["case_id"]: float(r["fc_diff"]) for r in csv.DictReader(open(os.path.join(SCREEN, "ideas10", "fc", "fc_sopbench.csv")))}
irt = {r[""]: float(r["b"]) for r in csv.DictReader(open(os.path.join(SCREEN, "..", "baseline", "data", "sopbench", "irt", "1d_1pl", "items.csv")))}
preds = {"sonnet_premortem (zero-shot, text)": fc, "public IRT b (28 public agents' runs)": irt,
         "refuse label (oracle-ish structure)": {c: float(not SHOULD[c]) for c in SAMPLE}}

def roc_auc_score(y, x):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); r = rankdata(x)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

q2 = {}
for pname, p in preds.items():
    ids = [c for c in SAMPLE if c in p]
    q2[pname] = {}
    # does the predictor even rank A-alone failures? (AUC of predicted difficulty for failure)
    for m, (a, b) in PAIRS.items():
        idm = [c for c in ids if c in D[m] and c in D[a] and c in D[b]]
        x = np.array([p[c] for c in idm])
        fail_a = np.array([not D[a][c]["success"] for c in idm], float)
        gain = np.array([float(D[m][c]["success"]) - float(D[a][c]["success"]) for c in idm])
        best = max((a, b), key=lambda k: np.mean([D[k][c]["success"] for c in idm]))
        gain_best = np.array([float(D[m][c]["success"]) - float(D[best][c]["success"]) for c in idm])
        auc = float(roc_auc_score(fail_a, x)) if 0 < fail_a.mean() < 1 else float("nan")
        # terciles of predicted difficulty (ties broken randomly)
        order = np.argsort(x + 1e-9 * rng.standard_normal(len(x)))
        terc = np.empty(len(x), int); terc[order] = np.arange(len(x)) * 3 // len(x)
        out = {"auc_pred_vs_A_fail": auc, "best_single": best}
        for t in range(3):
            sel = [idm[i] for i in range(len(idm)) if terc[i] == t]
            g = by_proc(sel, lambda c: float(D[m][c]["success"]) - float(D[a][c]["success"]))
            out[f"tercile{t} (0=easy) gain vs A alone"] = boot_ci(g)
            out[f"tercile{t} A-alone success"] = float(np.mean([D[a][c]["success"] for c in sel]))
        hi = [idm[i] for i in range(len(idm)) if terc[i] == 2]; lo = [idm[i] for i in range(len(idm)) if terc[i] < 2]
        out["gain hard third minus rest"] = float(np.mean([gain[idm.index(c)] for c in hi]) - np.mean([gain[idm.index(c)] for c in lo]))
        q2[pname][m] = out
res["q2_by_predicted_difficulty"] = q2

# Q2b: selective gate. Gate only the top x% predicted-hard cases; success and cost vs x (cross-run mix).
def sel_curve(m, a, p, fracs=(0, .1, .2, .3, .5, .7, 1.0)):
    idm = [c for c in SAMPLE if c in D[m] and c in D[a] and c in p]
    x = np.array([p[c] for c in idm]); order = list(np.argsort(-x))
    succ_pair = np.array([D[m][c]["success"] for c in idm], float); succ_a = np.array([D[a][c]["success"] for c in idm], float)
    usd_pair = np.array([D[m][c]["usd"] for c in idm]); usd_a = np.array([D[a][c]["usd"] for c in idm])
    oracle_gain = succ_pair - succ_a
    out = []
    for f in fracs:
        k = int(round(f * len(idm))); g = set(order[:k])
        s = np.mean([succ_pair[i] if i in g else succ_a[i] for i in range(len(idm))])
        u = np.sum([usd_pair[i] if i in g else usd_a[i] for i in range(len(idm))])
        # random choice of the same number of cases, expectation
        s_rand = succ_a.mean() + f * (succ_pair.mean() - succ_a.mean())
        og = np.sort(oracle_gain)[::-1][:k].sum() / len(idm) + succ_a.mean()
        out.append({"gated_share": f, "success": float(s), "success_random": float(s_rand), "success_oracle": float(og), "usd_150": float(u)})
    return out

res["q2_selective_gate"] = {f"{m} gated by {pn}": sel_curve(m, PAIRS[m][0], p) for m in ("flash>pro", "pro>flash") for pn, p in preds.items() if not pn.startswith("refuse")}

# ---------------- Q3: disagreement and errors ----------------
q3 = {}
# (a) at the gate: B deviates from A's proposal vs agrees
for m in PAIRS:
    for cls in (True, False):
        ids = [c for c in SAMPLE if (m, c) in GATE_CASE and SHOULD[c] == cls]
        for kind in ("agree", "deviate", "none"):
            sel = [c for c in ids if GATE_CASE[(m, c)] == kind]
            q3[f"{m} gate {kind} [{'perform' if cls else 'refuse'}]"] = {"n": len(sel), "success": float(np.mean([D[m][c]["success"] for c in sel])) if sel else None}
# (b) two independent single runs: do they agree on act / not act?
def indep(a, b, human_acc=1.0):
    ids = [c for c in SAMPLE if c in D[a] and c in D[b]]
    agree = [c for c in ids if D[a][c]["performed"] == D[b][c]["performed"]]
    dis = [c for c in ids if c not in agree]
    acc = lambda k, s: float(np.mean([D[k][c]["success"] for c in s])) if s else None
    # router: if they agree take model a's run, else human (correct with prob human_acc)
    succ = (sum(D[a][c]["success"] for c in agree) + human_acc * len(dis)) / len(ids)
    return {"n": len(ids), "agree_share": len(agree) / len(ids), f"{a} success when agree": acc(a, agree), f"{a} success when disagree": acc(a, dis),
            f"{b} success when agree": acc(b, agree), f"{b} success when disagree": acc(b, dis),
            "router agree->a, disagree->human(1.0)": float(succ), f"{a} alone": acc(a, ids), f"{b} alone": acc(b, ids)}
for a, b in (("flash", "pro"), ("flash", "g38"), ("pro", "g38"), ("lite", "g38"), ("flash", "flash (5 Oct)")):
    q3[f"independent {a} vs {b}"] = indep(a, b)
# (c) gate deviation as an escalation signal: send 'deviate' episodes to a human
for m in PAIRS:
    ids = [c for c in SAMPLE if (m, c) in GATE_CASE]
    dev = [c for c in ids if GATE_CASE[(m, c)] == "deviate"]
    succ = (sum(D[m][c]["success"] for c in ids if c not in dev) + len(dev)) / len(ids)
    q3[f"{m}: escalate gate disagreements to human"] = {"human_share": len(dev) / len(ids), "success": float(succ), "pair alone": float(np.mean([D[m][c]["success"] for c in ids]))}
res["q3_disagreement"] = q3

json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=str)

def pr(title, d):
    print("\n## " + title)
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{k}: " + ", ".join(f"{x}={(round(y, 3) if isinstance(y, float) else (tuple(round(z, 3) for z in y) if isinstance(y, tuple) else y))}" for x, y in v.items()))
        elif isinstance(v, list):
            print(k)
            for row in v: print("   ", {x: round(y, 3) for x, y in row.items()})
        else: print(k, v)

pr("Q1 cross-run flips", res["q1_cross_run"])
for m, t in res["q1_gate"].items(): pr(f"Q1 gate decisions {m}", t)
for pn, t in res["q2_by_predicted_difficulty"].items(): pr(f"Q2 by predicted difficulty: {pn}", t)
pr("Q2 selective gate", res["q2_selective_gate"])
pr("Q3 disagreement", res["q3_disagreement"])
