"""usage: run_llm.py <variant> <what> [limit]   variant A|B|C|INT ; what main|retest|samples"""
import sys, random, json
from concurrent.futures import ThreadPoolExecutor
import lib, prompts
import pandas as pd, numpy as np
BENCH = ["sopbench", "tau2"]
def units():
    """unique texts: {text: [case ids]}; domain per case"""
    U = {}; dom = {}
    for b in BENCH:
        st = lib.statements(b); t = pd.read_csv(lib.REPO / "data" / b / "tasks.csv").set_index("case_id")
        for c, tx in st.items(): U.setdefault((b, tx), []).append(c); dom[c] = (b, t.loc[c, "domain"], t.loc[c, "procedure_id"])
    return U, dom
def retest_set(U):
    keys = []
    for b, n, extra in [("sopbench", 30, 20), ("tau2", 15, 15)]:
        items = list(lib.statements(b).items()); random.Random(11).shuffle(items)
        first = [tx for _, tx in items[:n]]       # the sample of the repository retest (original ADeLe second pass exists)
        rest = [tx for _, tx in items[n:]]; random.Random(5).shuffle(rest)
        seen = set(first); add = []
        for tx in rest:
            if tx not in seen and len(add) < extra: add.append(tx); seen.add(tx)
        keys += [(b, tx) for tx in first + add]
    return list(dict.fromkeys(keys))
def anchors(U, dom):
    ad = pd.read_csv(lib.REPO / "data" / "sopbench" / "features" / "adele.csv").set_index("task_id")
    ad = pd.concat([ad, pd.read_csv(lib.REPO / "data" / "tau2" / "features" / "adele.csv").set_index("task_id")])
    ad["tot"] = ad[lib.DEMANDS].sum(1)
    texts = {}
    for b in BENCH: texts.update(lib.statements(b))
    uniq = {c[0] for (b, tx), c in U.items() if len(c) == 1} | {c[0] for (b, tx), c in U.items() if len(c) > 1}  # representative per text
    rep = {tx: c[0] for (b, tx), c in U.items()}; repset = set(rep.values())
    A = {}
    for (b, d) in sorted({(v[0], v[1]) for v in dom.values()}):
        ids = [c for c in ad.index if dom[c][:2] == (b, d) and c in repset]
        ids.sort(key=lambda c: (ad.loc[c, "tot"], c))
        def pick(qs, used):
            out = []
            for q in qs:
                i = int(q * (len(ids) - 1))
                for j in sorted(range(len(ids)), key=lambda j: abs(j - i)):
                    c = ids[j]
                    if c not in used and dom[c][2] not in {dom[o][2] for o in out}: out.append(c); used.add(c); break
            return out
        used = set(); s1 = pick([.1, .5, .9], used); s2 = pick([.25, .6, .8], used)
        A[(b, d)] = (s1, s2)
    return A, ad, texts
def main():
    var, what = sys.argv[1], sys.argv[2]; limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    U, dom = units(); keys = list(U)
    R = retest_set(U)
    jobs = []
    if var == "C": A, ad, texts = anchors(U, dom)
    def mk(b, tx):
        c = U[(b, tx)][0]
        if var == "A": return prompts.prompt_A(tx)
        if var == "B": return prompts.prompt_B(tx)
        if var == "INT": return prompts.prompt_INT(tx)
        if var == "C":
            s1, s2 = A[dom[c][:2]]; use = s2 if c in s1 else s1
            return prompts.prompt_C(tx, dom[c][1], [(texts[a], ad.loc[a].to_dict()) for a in use])
    if what == "main": sel = [(k, 0, 0.0) for k in keys]
    elif what == "retest": sel = [(k, 1, 0.0) for k in R]
    elif what == "samples": sel = [(k, r + 10, 0.7) for k in R for r in range(4)]
    if limit: sel = sel[:limit]
    before = lib._calls["n"]
    def one(a):
        (b, tx), rep, temp = a
        try: lib.gen(mk(b, tx), temp=temp, rep=rep)
        except Exception as e: print("ERR", str(e)[:100], flush=True)
    done = 0
    with ThreadPoolExecutor(6) as ex:
        for _ in ex.map(one, sel):
            done += 1
            if done % 50 == 0: print(done, len(sel), "calls", lib._calls["n"] - before, flush=True)
    print("finished", var, what, "calls", lib._calls["n"] - before)
if __name__ == "__main__": main()
