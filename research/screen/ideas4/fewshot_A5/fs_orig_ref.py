"""Few-shot plan + gold-length regression features for tau2 (fold-aware, training gold only)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, hashlib, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
S = ZR.SCREEN
REPO = Path(ZR.REPO)
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, S + "/ideas3/plan_tau"); sys.path.insert(0, S + "/ideas3/plan_tools")
import llm
import feats as F
HERE = Path(__file__).parent
PT = Path(S + "/ideas3/plan_tools")
TD = Path(ZR.DATA + "/tau2-domains/tasks")
CACHE = HERE / "llm_cache.jsonl"
lock = threading.Lock(); cache = {}
if CACHE.exists():
    for l in CACHE.read_text().splitlines():
        d = json.loads(l); cache[d["k"]] = d["r"]
NCALLS = [0]
TOOLS = json.load(open(PT / "tool_text.json")); TYPES = json.load(open(PT / "tool_types.json"))
STM = {}
for l in open(REPO / "data/tau2/statements.jsonl"):
    s = json.loads(l); STM[s["case_id"]] = s["text"]
EMB = np.load(REPO / "data/tau2/features/bge_base_en_v1.5.npz", allow_pickle=True)
EID = list(EMB["task_ids"]); EX = EMB["X"].astype(float); EX = EX / np.linalg.norm(EX, axis=1, keepdims=True); EIDX = {c: i for i, c in enumerate(EID)}
DOM = {c: c.split("/")[0] for c in STM}
# gold (private): accessible only through gold_of(case, allowed)
_GOLD = {}
_m = json.load(open(REPO / "data/tau2/id_map.json"))
for dom in ["airline", "retail", "telecom"]:
    tk = {str(t["id"]): t for t in json.load(open(TD / f"{dom}_tasks.json"))}
    for k, c in _m.items():
        if k.startswith(dom + ":"): _GOLD[c] = tk[k.split(":", 1)[1]]["evaluation_criteria"].get("actions") or []
def gold_of(case, allowed):
    assert case in allowed, f"gold of non-training task {case} requested"
    return _GOLD[case]
def gold_stats(acts, dom):
    ag = [a for a in acts if a.get("requestor", "assistant") != "user"]
    return [len(ag), sum(TYPES.get(f"{dom}/{a['name']}") == "write" for a in ag), len(acts)]

def call(prompt, tag="fs"):
    k = hashlib.sha256(f"0|{tag}|{prompt}".encode()).hexdigest()
    if k in cache: return cache[k]
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 8000, "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingLevel": "low"}}}
    data = json.dumps(body).encode(); text = None
    for attempt in range(8):
        url, auth = llm._endpoint(llm.MODEL)
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth})
        try:
            with urllib.request.urlopen(req, timeout=240) as r: d = json.loads(r.read())
            parts = ((d.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")); break
        except urllib.error.HTTPError as e:
            if e.code == 401: llm._TOKEN["exp"] = 0.0
            if e.code in (401, 429, 500, 502, 503, 504): time.sleep(min(60, 2 ** attempt + 1)); continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError): time.sleep(min(60, 2 ** attempt + 1))
    with lock:
        NCALLS[0] += 1
        if text is not None:
            cache[k] = text
            with open(CACHE, "a") as f: f.write(json.dumps({"k": k, "r": text}) + "\n")
    return text

import gen2
INSTR = gen2.INSTR.replace("=== CUSTOMER CASE ===", "{examples}=== CUSTOMER CASE ===", 1)
def render_gold(acts):
    out = []
    for a in acts:
        if a.get("requestor", "assistant") == "user": out.append(f'ask_customer_to_act(purpose="{a["name"]}")')
        else: out.append(f'{a["name"]}({json.dumps(a.get("arguments") or {}, separators=(",", ":"))[:200]})')
    return out
def build_prompt(case, train_set, k=5):
    dom = DOM[case]; cand = [c for c in train_set if DOM[c] == dom] or list(train_set)
    sims = EX[[EIDX[c] for c in cand]] @ EX[EIDX[case]]
    top = [cand[i] for i in np.argsort(-sims, kind="stable")[:k]]
    ex = ["=== SOLVED EXAMPLES: other cases with the reference sequence of tool calls (these come from a different case set) ===\n"]
    for j, c in enumerate(top, 1):
        acts = gold_of(c, train_set)
        ex.append(f"--- Example {j} (domain {DOM[c]}) ---\n{STM[c]}\nReference tool-call sequence ({len(acts)} steps):\n" + "\n".join(f"  {i + 1}. {x}" for i, x in enumerate(render_gold(acts))) + "\n")
    ex.append("=== END EXAMPLES ===\nNumber of steps and reads/writes in your plan should follow what the examples show for similar cases.\n\n")
    pol = gen2.policy(dom)
    return INSTR.format(policy=pol, tools=TOOLS[dom], case=STM[case], examples="".join(ex)), top

_B = {}
def fewshot_feats(tr, te, workers=8):
    """9 plan features per test task, prompts built from training tasks only."""
    tr_set = set(tr); assert not (tr_set & set(te))
    todo = [c for c in te if c not in _B or _B[c][0] != frozenset(tr_set)] if False else te
    def one(c):
        p, top = build_prompt(c, tr_set); assert all(t in tr_set for t in top) and c not in top
        txt = call(p); pl = F.parse(txt)
        return c, (F.feats(pl) if pl else None)
    with ThreadPoolExecutor(workers) as ex: res = dict(ex.map(one, te))
    return res
_AFEAT = ["plan_calls", "plan_writes", "plan_reads", "plan_distinct_tools", "plan_rules", "plan_facts", "plan_refuse", "plan_transfer", "plan_ambig"]
PLAN3 = pd.read_csv(PT / "features/plan3_tau2.csv").set_index("case_id")[_AFEAT]
PROC = pd.read_csv(REPO / "data/tau2/tasks.csv").set_index("case_id").procedure_id
def _ridge_cv(X, y, grp):
    m, s = X.mean(0), X.std(0) + 1e-9; Z = (X - m) / s; best, ba = 1e18, 1.0
    for a in [1, 10, 100, 1000]:
        mse = 0
        for i, j in GroupKFold(5).split(Z, groups=grp):
            mse += ((Ridge(alpha=a).fit(Z[i], y[i]).predict(Z[j]) - y[j]) ** 2).mean()
        if mse < best: best, ba = mse, a
    return m, s, ba
def stage_a(tr, te, X_all=None):
    """Ridge from plan features to gold [agent len, writes, all len] on training tasks. Train rows are out-of-fold."""
    X_all = PLAN3 if X_all is None else X_all
    tr = list(tr); tr_set = set(tr)
    Y = np.array([gold_stats(gold_of(c, tr_set), DOM[c]) for c in tr], float)
    X = X_all.loc[tr].to_numpy(float); grp = PROC.loc[tr].to_numpy()
    m, s, a = _ridge_cv(X, Y, grp)
    oof = np.zeros_like(Y)
    for i, j in GroupKFold(5).split(X, groups=grp):
        oof[j] = Ridge(alpha=a).fit((X[i] - m) / s, Y[i]).predict((X[j] - m) / s)
    full = Ridge(alpha=a).fit((X - m) / s, Y)
    pt = full.predict((X_all.loc[list(te)].to_numpy(float) - m) / s)
    cols = ["A_len", "A_writes", "A_alllen"]
    return pd.concat([pd.DataFrame(oof, index=tr, columns=cols), pd.DataFrame(pt, index=list(te), columns=cols)])

BCOLS = _AFEAT
def b_table(tr, te, scheme):
    """B features: few-shot plan features. Test tasks: examples from the training pool. Training tasks: examples from the training
    pool minus own procedure (new_procedures) or minus own procedure within own domain (new_domain)."""
    tr = list(tr); tr_set = set(tr); te = list(te); assert not (tr_set & set(te))
    def top_for(c, pool, k=5):
        cand = [x for x in pool if DOM[x] == DOM[c]] or list(pool)
        sims = EX[[EIDX[x] for x in cand]] @ EX[EIDX[c]]
        return [cand[i] for i in np.argsort(-sims, kind="stable")[:k]]
    jobs = []
    for c in te: jobs.append((c, top_for(c, tr)))
    for c in tr:
        pool = [x for x in tr if PROC[x] != PROC[c]] if scheme == "new_procedures" else [x for x in tr if DOM[x] == DOM[c] and PROC[x] != PROC[c]]
        jobs.append((c, top_for(c, pool)))
    def one(j):
        c, top = j; assert all(t in tr_set for t in top) and c not in top
        ex = ["=== SOLVED EXAMPLES: other cases with the reference sequence of tool calls (these come from a different case set) ===\n"]
        for n, t in enumerate(top, 1):
            acts = gold_of(t, tr_set)
            ex.append(f"--- Example {n} (domain {DOM[t]}) ---\n{STM[t]}\nReference tool-call sequence ({len(acts)} steps):\n" + "\n".join(f"  {i + 1}. {x}" for i, x in enumerate(render_gold(acts))) + "\n")
        ex.append("=== END EXAMPLES ===\nNumber of steps and reads/writes in your plan should follow what the examples show for similar cases.\n\n")
        p = INSTR.format(policy=gen2.policy(DOM[c]), tools=TOOLS[DOM[c]], case=STM[c], examples="".join(ex))
        pl = F.parse(call(p)); return c, (F.feats(pl) if pl else None)
    with ThreadPoolExecutor(4) as ex_: res = dict(ex_.map(one, jobs))
    df = pd.DataFrame.from_dict({c: (v or {}) for c, v in res.items()}, orient="index")[BCOLS]
    nfail = int(df.isna().any(axis=1).sum()); df = df.fillna(df.median())
    return df, nfail
