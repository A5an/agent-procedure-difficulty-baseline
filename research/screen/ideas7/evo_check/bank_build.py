"""Check 2: build F1 inputs for tau-Knowledge banking from documentation only. usage: bank_build.py test|run. Max 600 calls, cached."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, glob, hashlib, threading, os
S = ZR.SCREEN
sys.path.insert(0, S + "/ideas/dryrun")
import lib as DL   # sets Vertex env, gives generate_t, DEFS, FORMAT, validate_main, parse_json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
HERE = Path(S + "/ideas7/evo_check"); R = DL.REPO
BK = Path(S + "/ideas3/plan_tools/src_tau2/data/tau2/domains/banking_knowledge")
CACHE = HERE / "bank_cache.jsonl"; MAXCALLS = 600
lock = threading.Lock(); cache = {}; STATE = {"n": 0}
if CACHE.exists():
    for l in open(CACHE): d = json.loads(l); cache[d["k"]] = d["r"]
DOCS = [json.load(open(f)) for f in sorted(glob.glob(str(BK / "documents/*.json")))]
HEADER = (BK / "prompts/components/policy_header.md").read_text()
INDEX = "\n".join(f"D{i:03d}: {d['title']}" for i, d in enumerate(DOCS))
TOOLS = "\n".join(json.dumps(t.get("function", t), separators=(",", ":")) for t in json.load(open(HERE / "bank_tools.json")))
STM = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(R / "data/tauk_banking/statements.jsonl")}
vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit([d["title"] + " " + d["content"] for d in DOCS])
DM = vec.transform([d["title"] + " " + d["content"] for d in DOCS])
def retrieved(text, k=8):
    sims = (DM @ vec.transform([text]).T).toarray().ravel(); idx = np.argsort(-sims)[:k]
    return [DOCS[i] | {"n": int(i)} for i in idx]
def policy_block(text):
    docs = "\n\n".join(f"--- D{d['n']:03d} {d['title']} ---\n{d['content'][:6000]}" for d in retrieved(text))
    return (f"=== BANK POLICY ===\n{HEADER}\n=== END POLICY ===\n\n=== KNOWLEDGE BASE INDEX (titles of all documents the agent can search) ===\n{INDEX}\n=== END INDEX ===\n\n"
            f"=== KNOWLEDGE BASE DOCUMENTS MOST RELEVANT TO THIS CASE (what a search returns) ===\n{docs}\n=== END DOCUMENTS ===")
def call(prompt, temp, tag, think, maxtok, json_mode=False):
    k = hashlib.sha256(f"{temp}|{tag}|{think}|{prompt}".encode()).hexdigest()
    if k in cache: return cache[k]
    for attempt in range(3):
        with lock:
            if STATE["n"] >= MAXCALLS: raise SystemExit("call budget reached")
            STATE["n"] += 1
        try:
            out, fin = DL.generate_t(prompt, temperature=temp, max_tokens=maxtok, thinking=think)
        except Exception as e:
            out = None
        if out and out.strip():
            if tag == "dry":
                try:
                    if not DL.validate_main(DL.parse_json(out)): out = None
                except Exception: out = None
            else:
                try: json.loads(out[out.find("{"):out.rfind("}") + 1])
                except Exception: out = None
        if out: break
    with lock:
        if out:
            cache[k] = out
            with open(CACHE, "a") as f: f.write(json.dumps({"k": k, "r": out}) + "\n")
    return out
PLAN_INSTR = open(S + "/ideas3/plan_tools/gen2.py").read().split('INSTR = """')[1].split('"""')[0]
SC_INSTR = open(S + "/ideas4/user_scenario/gen.py").read().split('SC_INSTR = """')[1].split('"""')[0]
def jobs_for(cid):
    text = STM[cid]; pol = policy_block(text)
    plan_p = PLAN_INSTR.format(policy=pol, tools=TOOLS, case=text)
    sc_p = SC_INSTR.format(policy=pol, clauses=INDEX, case=text).replace("CLAUSE LIST (ids assigned to the policy above)", "KNOWLEDGE BASE DOCUMENT INDEX")
    sc_p += "\n(In 'clauses' list the document ids D### from the index that the case touches.)"
    dry_p = ("You are a compliance officer doing a dry run. Read the customer request next to the policy and judge, before touching "
             "any system, whether the request will have to be refused, which policy conditions cannot be checked without system data, "
             "and how much lookup work the agent will need.\n\n" + f"=== POLICY ===\n{pol}\n\n=== CASE ===\n{text}\n\n" + DL.DEFS + "\n\n" + DL.FORMAT)
    return [("plan_t0", plan_p, 0.0, "medium", 8000), ("plan_s1", plan_p, 1.0, "medium", 8000), ("plan_s2", plan_p, 1.0, "medium", 8000),
            ("scn", sc_p, 0.0, "medium", 8000), ("dry", dry_p, 0.0, "low", 6000)]
def run_case(cid):
    res = {}
    for tag, p, t, th, mt in jobs_for(cid):
        tg = "dry" if tag == "dry" else tag
        res[tag] = call(p, t, tg if tag != "plan_s2" else "plan_s2", th, mt)
    return cid, res
if __name__ == "__main__":
    ids = list(STM)
    if sys.argv[1] == "test":
        cid = ids[0]; js = jobs_for(cid); print(len(js[0][1]), "chars plan prompt"); print(js[3][1][-900:])
        c, r = run_case(cid); print({k: (v or "")[:700] for k, v in r.items()}); print("calls", STATE["n"])
    else:
        with ThreadPoolExecutor(6) as ex: out = dict(ex.map(run_case, ids))
        json.dump(out, open(HERE / "bank_raw.json", "w")); print("calls this process", STATE["n"], "missing", sum(v is None for d in out.values() for v in d.values()))
