import json, subprocess, sys, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent; S = HERE.parent
DOM = S.parent / "data/external/tauindianbank/code/data/tau2/domains/indian_banking"
POLICY = (DOM / "policy.md").read_text()
old, new = json.load(open(S / "tib/selection.json")), json.load(open(S / "combo_rep/selection_rep.json"))
purpose = {**old["purpose"], **new["purpose"]}
ASK = """A bank's virtual assistant will handle customer requests of the following type.

REQUEST TYPE (one-line description): {p}

BANK POLICY:
{pol}

Without seeing any real conversation: how much does handling requests of this type correctly depend on details that the
one-line description and the policy do not specify (which exact steps, in which order, which final action, how the
customer behaves, edge cases, conventions of this bank)? Answer JSON only:
{{"gap_score": <0-10>, "gaps": ["...", "..."]}}"""


def ask(f):
    out = HERE / "answers" / f"{f}.json"
    if out.exists(): return f, json.load(open(out))
    r = subprocess.run(["claude", "-p", "--model", "sonnet", "--tools", "", "--setting-sources", "", "--no-session-persistence",
                        "--output-format", "json"], input=ASK.format(p=purpose[f], pol=POLICY), capture_output=True, text=True, timeout=600, cwd="/tmp")
    t = json.loads(r.stdout)["result"]; j = json.loads(t[t.find("{"): t.rfind("}") + 1])
    out.parent.mkdir(exist_ok=True); json.dump(j, open(out, "w"), indent=1)
    return f, j


with ThreadPoolExecutor(4) as ex:
    G = dict(ex.map(ask, list(purpose)))


def rates(path, fams):
    d = collections.defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        if r.get("reward") is None: continue
        f = next((f for f in fams if f"_{f}_" in r["task_id"]), None)
        if f: d[f].append(1 - int(r["reward"] >= 1))
    return {k: np.mean(v) for k, v in d.items()}


err_doc, err_seed = collections.defaultdict(list), collections.defaultdict(list)
for base, fams, dn in ((S / "tib", old["test_families"], "probes_{a}.v2.jsonl"), (S / "combo_rep", new["test_families"], "doc_{a}.jsonl")):
    for a in ("g38", "flash"):
        re_, do, se = rates(base / f"runs/real_{a}.jsonl", fams), rates(base / "runs" / dn.format(a=a), fams), rates(base / f"runs/seeded_{a}.jsonl", fams)
        for f in fams:
            err_doc[f].append(abs(do[f] - re_[f])); err_seed[f].append(abs(se[f] - re_[f]))
fams = sorted(G)
g = np.array([float(G[f]["gap_score"]) for f in fams]); ed = np.array([np.mean(err_doc[f]) for f in fams])
gain = np.array([np.mean(err_doc[f]) - np.mean(err_seed[f]) for f in fams])
rng = np.random.default_rng(1)
for name, yv in (("doc-probe error", ed), ("gain of seeded over doc probes", gain)):
    r = spearmanr(g, yv).correlation
    bs = [spearmanr(g[i], yv[i]).correlation for i in (rng.integers(0, len(g), len(g)) for _ in range(4000)) if np.ptp(g[i]) and np.ptp(yv[i])]
    print(f"gap score vs {name}: rho {r:+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
print("gap scores:", sorted(collections.Counter(g.astype(int)).items()))
