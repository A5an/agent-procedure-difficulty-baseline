"""Four case-level L0 families for SOPBench. usage: feats.py  -> fam_nli.csv fam_ret.csv fam_act.csv fam_bnd.csv nli_mats.pkl"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, re, pickle, numpy as np, pandas as pd
S = ZR.SCREEN
sys.path.insert(0, S + "/ideas"); sys.path.insert(0, ".")
import prep
from cpulock import cpu_lock
C = prep.CASES; ids = list(C)
# ---------------- family 1: NLI (static part; skip-weighted part built per fold in run.py)
def fam_nli():
    R = pickle.load(open("nli_cache.pkl", "rb"))["res"]   # order: entailment, neutral, contradiction
    rows, mats = [], {}
    for c in ids:
        v = C[c]; cs = v["conds"]
        # M[k, j, :] probabilities for condition k, premise j (last premise = whole message)
        M = np.array([[R[(s, cd["text"])] for s in v["sents"] + [v["msg"]]] for cd in cs])  # K x (S+1) x 3
        mats[c] = M
        Ms = M[:, :-1, :]
        ent_k = Ms[:, :, 0].max(1); con_k = Ms[:, :, 2].max(1); full_ent = M[:, -1, 0]; full_con = M[:, -1, 2]
        silent = (np.maximum(ent_k, con_k) < 0.5)
        rows.append(dict(case_id=c, nli_ent_max=ent_k.max(), nli_ent_mean=ent_k.mean(), nli_con_max=con_k.max(), nli_con_mean=con_k.mean(),
                         nli_n_silent=silent.sum(), nli_frac_silent=silent.mean(), nli_full_ent=full_ent.max(), nli_full_con=full_con.max()))
    pd.DataFrame(rows).set_index("case_id").to_csv("fam_nli.csv"); pickle.dump(mats, open("nli_mats.pkl", "wb"))
# ---------------- family 2: retrieval margin (bge-base on CPU)
def fam_ret():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    texts = sorted({cd["text"] for v in C.values() for cd in v["conds"]})
    E = dict(zip(texts, m.encode(texts, batch_size=32, normalize_embeddings=True)))
    qs = {c: "Represent this sentence for searching relevant passages: " + v["msg"] + " Requested action: " + v["act"].replace("_", " ") for c, v in C.items()}
    Q = dict(zip(qs, m.encode(list(qs.values()), batch_size=32, normalize_embeddings=True)))
    dom_texts = {}
    for v in C.values(): dom_texts.setdefault(v["domain"], set()).update(cd["text"] for cd in v["conds"])
    rows = []
    for c in ids:
        v = C[c]; own = [cd["text"] for cd in v["conds"]]; os_ = set(own)
        pool = [t for t in dom_texts[v["domain"]] if t not in os_]
        q = Q[c]; so = np.array([E[t] @ q for t in own])
        sp = np.array([E[t] @ q for t in pool]) if pool else np.array([0.0])
        best = sp.max(); marg = so - best
        rank = np.array([(sp > s).sum() + 1 for s in so]) / (len(sp) + 1)   # 1/(n+1) is best
        rows.append(dict(case_id=c, ret_sim_mean=so.mean(), ret_sim_min=so.min(), ret_margin_min=marg.min(), ret_margin_mean=marg.mean(),
                         ret_rank_worst=rank.max(), ret_rank_mean=rank.mean(), ret_share_low=(marg < 0).mean(), ret_best_distr=best))
    pd.DataFrame(rows).set_index("case_id").to_csv("fam_ret.csv")
# ---------------- family 3: action readiness (heuristic, statement text only)
NUMRX = re.compile(r"(?<![\w.])\$?\d[\d,]*(?:\.\d+)?%?")
IDTOK = re.compile(r"\b(?:[A-Za-z]+_[A-Za-z0-9_]+|(?=[A-Za-z0-9\-]*[A-Za-z])(?=[A-Za-z0-9\-]*\d)[A-Za-z0-9\-]{4,}|\d{4}(?: \d{4}){3})\b")
BADNAME = {"Type", "Time", "Date", "Number", "Name", "Amount", "Description", "Address", "Password", "Score", "You", "Regards", "Thanks", "Id", "ID", "Details", "Request"}
NAMERX = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
DATERX = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T[\d:]+)?|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
def kind(name, typ):
    n = name.lower()
    if "password" in n or n.startswith("identification") or n in ("vin",): return "secret"
    if "date" in n or n.endswith("time"): return "date"
    if typ in ("float", "int") or any(k in n for k in ("amount", "income", "assets", "limit", "score", "points")): return "num"
    if any(k in n for k in ("username", "guest_name", "name", "owner", "card_number", "plate", "_id", "provider", "claim")): return "id"
    return "enum"
def fam_act():
    rows = []
    for c in ids:
        v = C[c]; args = prep.TA.get((v["domain"], v["act"]), []); m = v["msg"]; ml = m.lower()
        dts = len(DATERX.findall(m)); md = DATERX.sub(" ", m); toks = set(IDTOK.findall(md)); md2 = IDTOK.sub(" ", md)
        nums = len(NUMRX.findall(md2)); idt = len(toks) + len([x for x in NAMERX.findall(md2) if x.split()[1] not in BADNAME])
        ks = [kind(a, t) for a, t in args]; K = {k: ks.count(k) for k in ("num", "id", "date", "secret", "enum")}
        cue = [int(all(w in ml for w in a.lower().replace("_new", "").split("_")[:1]) or a.lower().replace("_", " ") in ml) for a, _ in args]
        miss_num = max(0, K["num"] - nums); miss_id = max(0, K["id"] - idt); miss_date = max(0, K["date"] - dts)
        n = max(len(args), 1)
        rows.append(dict(case_id=c, act_n_args=len(args), act_n_secret=K["secret"], act_n_enum=K["enum"], act_miss_num=miss_num, act_miss_id=miss_id, act_miss_date=miss_date,
                         act_frac_present=1 - (miss_num + miss_id + miss_date + K["secret"] * (1 - int(bool(re.search(r"password|pin|identification|license|passport", ml))))) / n,
                         act_frac_cue=sum(cue) / n, act_nums=nums, act_idtok=idt))
    pd.DataFrame(rows).set_index("case_id").to_csv("fam_act.csv")
# ---------------- family 4: boundary margin
CMP = re.compile(r"at least|at most|no more than|no less than|more than|less than|exceed|higher|lower|greater|fewer|below|above|over|under|minimum|maximum|within|up to|not (?:be )?(?:more|less)|limit|threshold|older|younger|years|days|hours|months|equal", re.I)
def thr_nums(t):
    out = []
    for x in NUMRX.findall(t):
        s = x.replace("$", "").replace(",", "").replace("%", "")
        try: out.append(float(s))
        except: pass
    return out
def fam_bnd():
    rows = []
    for c in ids:
        v = C[c]
        th = [x for cd in v["conds"] if CMP.search(cd["text"]) for x in thr_nums(re.sub(r"\b(?:ANY ONE|ALL)\b", "", cd["text"]))]
        th = [x for x in th if x != 0]
        ms = []
        for x in NUMRX.findall(v["msg"]):
            try: ms.append(float(x.replace("$", "").replace(",", "").replace("%", "")))
            except: pass
        pr = [(abs(x - t) / max(abs(t), 1.0), x - t) for x in ms for t in th if x > 0]
        if pr:
            mn = min(p[0] for p in pr)
            rows.append(dict(case_id=c, bnd_has=1.0, bnd_logmargin=np.log10(mn + 1e-3), bnd_onthr=float(any(p[0] == 0 for p in pr)), bnd_near10=float(mn <= 0.10),
                             bnd_frac_above=float(np.mean([p[1] > 0 for p in pr])), bnd_npairs=len(pr)))
        else:
            rows.append(dict(case_id=c, bnd_has=0.0, bnd_logmargin=np.nan, bnd_onthr=0.0, bnd_near10=0.0, bnd_frac_above=np.nan, bnd_npairs=0))
    pd.DataFrame(rows).set_index("case_id").to_csv("fam_bnd.csv")
if __name__ == "__main__":
    which = sys.argv[1:]
    with cpu_lock("case_signals_feats"):
        if "act" in which: fam_act()
        if "bnd" in which: fam_bnd()
        if "ret" in which: fam_ret()
        if "nli" in which: fam_nli()
