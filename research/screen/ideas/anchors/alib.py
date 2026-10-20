"""In-context anchors: prompt construction, cache, predictor. Shared by precompute.py and run.py."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import hashlib, json, os, re, sys, threading
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, ZR.SCREEN)
sys.path.insert(0, ZR.SCREEN + "/ideas")
import harness as H  # noqa: E402

REPO = H.REPO
DATA = REPO / "data"
POL = Path(ZR.DATA + "/tau2-domains/tasks")
POLICY = {"airline": "airline_policy.md", "retail": "retail_policy.md", "telecom": "telecom_main_policy.md"}
HERE = Path(__file__).parent
CACHE = HERE / "llm_cache.jsonl"
N_NEAR, N_Q = 4, 4

_lock = threading.Lock()
_mem = None


def load_cache():
    global _mem
    if _mem is None:
        _mem = {}
        if CACHE.exists():
            for l in CACHE.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    _mem[r["h"]] = r["text"]
    return _mem


def phash(p):
    return hashlib.sha256(p.encode()).hexdigest()


def cache_put(h, text):
    with _lock:
        load_cache()[h] = text
        with open(CACHE, "a") as f:
            f.write(json.dumps({"h": h, "text": text}) + "\n")


_stm = {}
def statements(bench):
    if bench not in _stm:
        _stm[bench] = {}
        for l in open(DATA / bench / "statements.jsonl"):
            r = json.loads(l)
            _stm[bench][r["case_id"]] = r["text"]
    return _stm[bench]


_emb = {}
def embeddings(bench):
    if bench not in _emb:
        d = np.load(DATA / bench / "features" / "bge_base_en_v1.5.npz", allow_pickle=True)
        X = d["X"].astype(np.float64)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        _emb[bench] = (dict(zip(d["task_ids"].tolist(), range(len(X)))), X)
    return _emb[bench]


def shares_of(data, train):
    S = {t: 0.0 for t in train}; N = {t: 0.0 for t in train}
    ts = set(train)
    for a in data.train_abilities.index:
        for t, v in data.responses.get(a, {}).items():
            if t in ts:
                k, n = (v["successes"], v["trials"]) if isinstance(v, dict) else (int(v), 1)
                S[t] += k; N[t] += n
    return {t: (S[t] + 0.5) / (N[t] + 1.0) for t in train}


def fit_map(data, train, share):
    x = np.array([-logit(share[t]) for t in train])
    y = data.train_items.loc[train, "b"].values.astype(float)
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def pick_anchors(bench, train, test, beta, mode="sim", seed=0):
    idx, X = embeddings(bench)
    tr = np.array([idx[t] for t in train])
    bt = np.array([beta[t] for t in train])
    q = np.quantile(bt, [0.25, 0.5, 0.75])
    qi = np.digitize(bt, q)  # 0..3
    out = {}
    for t in test:
        sim = X[tr] @ X[idx[t]]
        if mode == "random":
            rng = np.random.default_rng(int(hashlib.md5(f"{t}|{seed}".encode()).hexdigest()[:8], 16))
            ch = list(rng.choice(len(train), N_NEAR + N_Q, replace=False))
            out[t] = ([train[i] for i in ch[:4]], [train[i] for i in ch])
            continue
        order = np.argsort(-sim, kind="stable")
        near = list(order[:N_NEAR])
        chosen = list(near)
        for k in range(4):
            for i in order:
                if qi[i] == k and i not in chosen:
                    chosen.append(i); break
        out[t] = ([train[i] for i in near], [train[i] for i in chosen])
    return out


def build_prompt(bench, tdom, target, anchors, share, dom_of):
    st = statements(bench)
    parts = []
    if bench == "tau2":
        pol = (POL / POLICY[tdom]).read_text()
        parts.append(f"The agents were given this domain policy (domain of the TARGET case: {tdom}):\n<policy>\n{pol}\n</policy>\n")
    parts.append("Below are cases from the same benchmark with the share of AI agent runs that solved each of them, "
                 "followed by a new target case.\n")
    for i, a in enumerate(sorted(anchors, key=lambda a: share[a]), 1):
        parts.append(f"### Example {i} (domain: {dom_of[a]})\n{st[a]}\nSolved in {share[a] * 100:.0f}% of agent runs.\n")
    parts.append(f"### Target case (domain: {tdom})\n{st[target]}\n")
    parts.append("Estimate the percentage of agent runs that solve the target case. "
                 'Answer with JSON only: {"success_percent": <number from 0 to 100>}')
    return "\n".join(parts)


def parse(text):
    m = re.search(r'"success_percent"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text or "")
    return float(m.group(1)) if m else None


def fold_prompts(bench, tasks_df, data, mode="sim", seed=0, only=None):
    train = list(data.train_tasks); test = list(data.test_tasks)
    if only is not None:
        test = [t for t in test if t in only]
    share = shares_of(data, train)
    beta = data.train_items.loc[train, "b"].to_dict()
    anc = pick_anchors(bench, train, test, beta, mode, seed)
    dom_of = dict(zip(tasks_df.case_id, tasks_df.domain))
    prompts = {t: build_prompt(bench, dom_of[t], t, anc[t][1], share, dom_of) for t in test}
    return train, test, share, beta, anc, prompts


class AnchorPredictor:
    """mode: llm | knn4 | random. The LLM answers are read from the cache (filled by precompute.py)."""
    def __init__(self, bench, mode="llm", name="anchors"):
        self.bench, self.mode, self.name = bench, mode, name
        self.tasks = pd.read_csv(DATA / bench / "tasks.csv")
        self._predicted_difficulties = {}
        self.n_fallback = 0
        self.raw = {}

    def fit(self, data, train_task_ids):
        self._p, self._predicted_difficulties = {}, {}
        sim_mode = "random" if self.mode == "random" else "sim"
        train, test, share, beta, anc, prompts = fold_prompts(self.bench, self.tasks, data, sim_mode)
        a, b = fit_map(data, train, share)
        cache = load_cache()
        for t in test:
            near = anc[t][0]
            knn_b = float(np.mean([beta[x] for x in near]))
            if self.mode == "knn4":
                bh = knn_b
            else:
                r = cache.get(phash(prompts[t]))
                v = parse(r)
                if v is None:
                    if r is None:
                        raise RuntimeError("missing LLM answer in cache; run precompute.py")
                    self.n_fallback += 1
                    bh = knn_b
                else:
                    p = min(max(v, 1.0), 99.0) / 100
                    bh = a + b * (-logit(p))
                    self.raw[t] = v
            self._predicted_difficulties[t] = float(bh)
        self.agents = list(data.train_abilities.index)
        self.theta = data.train_abilities["ability"].to_dict()

    def predict_probability(self, data, agent_id, task_id):
        return float(expit(self.theta[agent_id] - self._predicted_difficulties[task_id]))
