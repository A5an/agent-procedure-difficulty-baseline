"""Pre-mortem ensemble sources S2 (Sonnet P2), G1 (Gemini P1), G2 (Gemini P2). usage: fcx.py gemini|sonnet [test]"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, random, hashlib, threading, time, subprocess, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "common")); sys.path.insert(0, str(HERE.parent / "fc")); sys.path.insert(0, ZR.SCREEN + "/ideas9/gap")
import vx
from gap import inputs
import fc as FC1
P2 = """You are the QA lead of a team that has read thousands of transcripts of tool-using LLM customer-service agents of the 2024-2025 generation (GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet, Gemini 1.5/2.0, Llama 3 70B, o1-mini) on strictly graded benchmarks. In those benchmarks the agent talks to the customer over several turns, calls tools against a company database it cannot see in advance, and passes only if it follows the policy exactly (every required check and confirmation, in order, the right final action or refusal, and a correct final database state).

Estimate what share of such agents would pass THIS case.

=== POLICY ===
{policy}
=== END POLICY ===

=== TOOLS ===
{tools}
=== END TOOLS ===

=== CUSTOMER REQUEST OR SCENARIO ===
{request}
=== END ===

First list the steps the agent must get exactly right in this case. Then list the traps where agents of that generation typically go wrong in a case like this, each with the probability that a typical agent falls into it. Return ONLY a JSON object:
{{"steps": ["short phrase"], "failure_modes": [{{"what": "short, specific to this case", "p": probability 0 to 1}}], "p_success": probability 0 to 1 that a typical agent of that generation passes this case}}"""
def p2(it): return P2.format(policy=it[2], tools=it[3], request=it[4])
C = vx.Client(HERE)
if __name__ == "__main__":
    which = sys.argv[1]; items = inputs()
    if len(sys.argv) > 2: items = random.Random(5).sample(items, 4)
    if which == "gemini":
        jobs = []
        for it in items:
            jobs.append((FC1.prompt_of(it), "G1")); jobs.append((p2(it), "G2"))
        u = list(dict.fromkeys(jobs))
        with ThreadPoolExecutor(int(os.environ.get("NT", "24"))) as ex: list(ex.map(lambda j: C.call(j[0], model="gemini-3.8-flash", tag=j[1]), u))
        res = {"G1": {it[1]: C.call(FC1.prompt_of(it), model="gemini-3.8-flash", tag="G1") for it in items},
               "G2": {it[1]: C.call(p2(it), model="gemini-3.8-flash", tag="G2") for it in items}}
        json.dump(res, open(HERE / ("gem_test.json" if len(sys.argv) > 2 else "gem_raw.json"), "w"))
        print("done gemini", C.ncalls, "fails", C.nfail)
    else:
        FC1.CACHE = HERE / "sonnet_cache.jsonl"; FC1.cache.clear()
        if FC1.CACHE.exists():
            for l in FC1.CACHE.read_text().splitlines(): d = json.loads(l); FC1.cache[d["k"]] = d["r"]
        FC1.WD = HERE / "work"
        order = [i for i in items if i[0] != "sopbench"] + [i for i in items if i[0] == "sopbench"]
        ps = {}
        for it in order: ps.setdefault(p2(it), []).append(it[1])
        keys = list(ps)
        with ThreadPoolExecutor(int(os.environ.get("NW", "12"))) as ex: outs = list(ex.map(FC1.call, keys))
        res = {c: o for p, o in zip(keys, outs) for c in ps[p]}
        json.dump(res, open(HERE / ("son_test.json" if len(sys.argv) > 2 else "son_raw.json"), "w"))
        print("done sonnet", len(res), "missing", sum(v is None for v in res.values()))
