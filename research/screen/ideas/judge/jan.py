import json, re, sys, hashlib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
rows = json.load(open("sample.json")); rng = np.random.default_rng(0)
def parse(t):
    m = re.search(r"\{.*\}", t, re.S)
    if not m: return None
    try: d = json.loads(m.group(0)); return bool(d["correct"]), float(d.get("confidence", np.nan)), str(d.get("reason", ""))
    except Exception:
        c = re.search(r'"correct"\s*:\s*(true|false)', t); 
        return (c.group(1) == "true", np.nan, "") if c else None
def load(think):
    c = {}
    for l in open(f"cache_{think}.jsonl"): d = json.loads(l); c[d["key"]] = d
    out = []
    for r in rows:
        k = hashlib.sha256((think + r["prompt"]).encode()).hexdigest()
        if k in c:
            p = parse(c[k]["text"])
            if p: out.append(dict(id=r["id"], bench=r["bench"], agent=r["agent"], case_id=r["case_id"], domain=r["domain"], grader=r["outcome"],
                                  should_succeed=r["should_succeed"], judge=int(p[0]), conf=p[1], reason=p[2]))
    return pd.DataFrame(out)
def stats(df):
    g, j = df.grader.values, df.judge.values
    acc = (g == j).mean(); fp = ((j == 1) & (g == 0)).sum() / max((g == 0).sum(), 1); fn = ((j == 0) & (g == 1)).sum() / max((g == 1).sum(), 1)
    po = acc; pe = (g.mean() * j.mean() + (1 - g.mean()) * (1 - j.mean())); kap = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return dict(n=len(df), acc=acc, fpr=fp, fnr=fn, kappa=kap, bal=1 - (fp + fn) / 2)
def boot(df, B=2000):
    base = stats(df); arr = {k: [] for k in ("acc", "fpr", "fnr", "kappa")}
    a = df[["grader", "judge"]].values
    for _ in range(B):
        s = pd.DataFrame(a[rng.integers(0, len(a), len(a))], columns=["grader", "judge"]); st = stats(s)
        for k in arr: arr[k].append(st[k])
    return base, {k: np.nanpercentile(v, [2.5, 97.5]) for k, v in arr.items()}
def fmt(df, name):
    b, ci = boot(df)
    return f"| {name} | {b['n']} | {b['acc']:.3f} [{ci['acc'][0]:.3f}, {ci['acc'][1]:.3f}] | {b['fpr']:.3f} [{ci['fpr'][0]:.3f}, {ci['fpr'][1]:.3f}] | {b['fnr']:.3f} [{ci['fnr'][0]:.3f}, {ci['fnr'][1]:.3f}] | {b['kappa']:.3f} [{ci['kappa'][0]:.3f}, {ci['kappa'][1]:.3f}] |"
H = "| group | n | accuracy | FPR (judge correct, grader fail) | FNR (judge wrong, grader pass) | kappa |\n|---|---|---|---|---|---|"
def report(think, sub=None):
    df = load(think)
    if sub is not None: df = df[df.id.isin(sub)]
    L = [f"### Judge thinking={think}, parsed {len(df)} verdicts", H, fmt(df, "all")]
    for b in ("sopbench", "tau2"): L.append(fmt(df[df.bench == b], b))
    for (b, a), g in df.groupby(["bench", "agent"]): L.append(fmt(g, f"{b} / {a}"))
    return df, "\n".join(L)
if __name__ == "__main__":
    df, txt = report("low"); print(txt)
    st = stats(df); e_sym = 1 - st["acc"]; print("sample acc", st)
    # symmetric e = 1 - balanced accuracy; also by bench
    for b in ("sopbench", "tau2"): s = stats(df[df.bench == b]); print(b, "e_sym(bal)", round((s['fpr'] + s['fnr']) / 2, 3), "fpr", round(s['fpr'], 3), "fnr", round(s['fnr'], 3))
    s = df[df.bench == "sopbench"].copy(); s["type"] = np.where(s.should_succeed == 1, "perform", "refuse")
    print(s.groupby("type").apply(lambda g: pd.Series(dict(n=len(g), acc=(g.grader == g.judge).mean(), grader_pass=g.grader.mean(), judge_correct=g.judge.mean(), fpr=((g.judge == 1) & (g.grader == 0)).sum() / max((g.grader == 0).sum(), 1), fnr=((g.judge == 0) & (g.grader == 1)).sum() / max((g.grader == 1).sum(), 1)))))
    d = df.dropna(subset=["conf"]).copy(); d["ok"] = (d.grader == d.judge).astype(int)
    print("conf AUC (conf -> judge agrees with grader)", roc_auc_score(d.ok, d.conf) if d.ok.nunique() > 1 else None, "mean conf ok/wrong", d[d.ok == 1].conf.mean(), d[d.ok == 0].conf.mean(), "n wrong", (d.ok == 0).sum())
    for b in ("sopbench", "tau2"):
        x = d[d.bench == b]; print(b, "AUC", roc_auc_score(x.ok, x.conf))
    # probabilistic: P(grader pass) from judge: conf if judge=1 else 1-conf
    d["p"] = np.where(d.judge == 1, d.conf, 1 - d.conf); print("AUC of judge P(correct) vs grader pass", roc_auc_score(d.grader, d.p))
