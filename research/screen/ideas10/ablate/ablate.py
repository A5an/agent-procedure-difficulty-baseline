"""Ablation calls D (direct rating) and R (retest of P1). usage: ablate.py"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "fc")); sys.path.insert(0, str(HERE.parent / "fresh")); sys.path.insert(0, ZR.SCREEN + "/ideas9/gap")
import fc as FC1
import fcg as G
from gap import inputs
ASK = "\n\nEstimate the probability that a typical agent of that generation fully succeeds on this {what}. Return ONLY a JSON object, no prose: {{\"p_success\": probability between 0 and 1}}"
def d_tau(it):
    p = FC1.prompt_of(it).replace("You are doing a pre-mortem. ", "")
    intro, rest = p.split("\n\nAssume the agent WILL fail.", 1)
    body = rest[rest.index("=== POLICY ==="):rest.index("=== END ===") + len("=== END ===")]
    return intro + "\n\n" + body + ASK.format(what="case")
def d_gen(it):
    p = G.prompt(it).replace("You are doing a pre-mortem. ", "")
    intro, rest = p.split("\n\nAssume the agent WILL fail.", 1)
    body = rest[rest.index("=== TASK ==="):rest.index("=== END TASK ===") + len("=== END TASK ===")]
    return intro + "\n\n" + body + ASK.format(what="task")
R_SUF = "\n\n(Independent second assessment.)"
tau = [i for i in inputs() if i[0] != "sopbench"]; gen = [i for i in G.items if i["bench"] in ("tac", "mcpmark")]
sop = [i for i in inputs() if i[0] == "sopbench"]; dra = [i for i in G.items if i["bench"] == "drafter"]
jobs = {}
for it in tau:
    jobs[("D", it[1])] = d_tau(it); jobs[("R", it[1])] = FC1.prompt_of(it) + R_SUF
for it in gen:
    k = f'{it["bench"]}::{it["id"]}'; jobs[("D", k)] = d_gen(it); jobs[("R", k)] = G.prompt(it) + R_SUF
for it in sop: jobs[("D", it[1])] = d_tau(it)
for it in dra: jobs[("D", f'{it["bench"]}::{it["id"]}')] = d_gen(it)
if __name__ == "__main__":
    FC1.CACHE = HERE / "sonnet_cache.jsonl"; FC1.cache.clear(); FC1.WD = HERE / "work"
    if FC1.CACHE.exists():
        for l in FC1.CACHE.read_text().splitlines(): d = json.loads(l); FC1.cache[d["k"]] = d["r"]
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        k = list(jobs)[0]; print(jobs[k][-900:]); k = [x for x in jobs if x[0] == "D" and "::" in x[1]][0]; print("-----"); print(jobs[k][:300], "...", jobs[k][-500:]); sys.exit()
    uniq = list(dict.fromkeys(jobs.values()))
    with ThreadPoolExecutor(int(os.environ.get("NW", "10"))) as ex: list(ex.map(FC1.call, uniq))
    json.dump({f"{a}|{b}": FC1.cache.get(__import__("hashlib").sha256(p.encode()).hexdigest()) for (a, b), p in jobs.items()}, open(HERE / "ablate_raw.json", "w"))
    print("done", len(jobs))
