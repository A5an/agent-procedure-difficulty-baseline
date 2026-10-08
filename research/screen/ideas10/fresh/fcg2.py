"""Round-2 sources on fresh benchmarks: SP2 (Sonnet P2), GP2 (Gemini P2), OP1 (Opus P1). usage: fcg2.py sonnet|gemini|opus"""
import sys, json, os, subprocess, hashlib, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "common")); sys.path.insert(0, str(HERE.parent / "fc")); sys.path.insert(0, str(HERE))
import vx
import fcg as G
import fc as FC1
P2 = """You are the QA lead of a team that has read thousands of transcripts of tool-using LLM agents of the 2024-2025 generation (GPT-4o, GPT-4.1, o3/o4-mini, Claude 3.5 to 4 Sonnet, Gemini 1.5 to 2.5, Llama 3 70B/405B, Qwen 2.5/3, DeepSeek V3, Kimi K2) on strictly graded agent benchmarks.

Environment and grading of this benchmark: {env}

Estimate what share of such agents would fully pass THIS task.

=== TASK ===
{text}
=== END TASK ===

First list the steps the agent must get exactly right in this task. Then list the traps where agents of that generation typically go wrong in a task like this, each with the probability that a typical agent falls into it. Return ONLY a JSON object:
{{"steps": ["short phrase"], "failure_modes": [{{"what": "short, specific to this task", "p": probability 0 to 1}}], "p_success": probability 0 to 1 that a typical agent of that generation fully passes this task}}"""
def p2(it): return P2.format(env=G.ENV[it["bench"]], text=it["text"])
items = [i for i in G.items if i["bench"] in ("tac", "mcpmark")] + [i for i in G.items if i["bench"] == "drafter"]
if __name__ == "__main__":
    w = sys.argv[1]
    if w == "gemini":
        C = vx.Client(HERE)
        with ThreadPoolExecutor(24) as ex: list(ex.map(lambda it: C.call(p2(it), model="gemini-3.8-flash", tag="GP2"), items))
        json.dump({f'{it["bench"]}::{it["id"]}': C.call(p2(it), model="gemini-3.8-flash", tag="GP2") for it in items}, open(HERE / "gp2_raw.json", "w")); print("done gemini")
    else:
        FC1.CACHE = HERE / f"{w}_cache2.jsonl"; FC1.cache.clear(); FC1.WD = HERE / "work"
        if FC1.CACHE.exists():
            for l in FC1.CACHE.read_text().splitlines(): d = json.loads(l); FC1.cache[d["k"]] = d["r"]
        if w == "opus":
            orig = FC1.call
            def call_opus(prompt):
                k = hashlib.sha256(prompt.encode()).hexdigest()
                if k in FC1.cache: return FC1.cache[k]
                cmd = ["claude", "-p", "--model", "opus", "--tools", "", "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
                text = None
                for a in range(5):
                    try: res = json.loads(subprocess.run(cmd, input=prompt, cwd=FC1.WD, capture_output=True, text=True, timeout=900).stdout)
                    except Exception as e: res = {"is_error": True, "result": str(e)}
                    if not res.get("is_error") and res.get("result"): text = res["result"]; break
                    time.sleep(30 * (a + 1))
                if text:
                    with FC1.lock:
                        FC1.cache[k] = text
                        with open(FC1.CACHE, "a") as f: f.write(json.dumps({"k": k, "r": text}) + "\n")
                return text
            fn, pf = call_opus, (lambda it: G.prompt(it))
        else:
            fn, pf = FC1.call, p2
        its = [i for i in items if i["bench"] != "drafter"] if w == "opus" else items
        with ThreadPoolExecutor(int(os.environ.get("NW", "8"))) as ex: outs = list(ex.map(lambda it: fn(pf(it)), its))
        json.dump({f'{it["bench"]}::{it["id"]}': o for it, o in zip(its, outs)}, open(HERE / f"{w}2_raw.json", "w")); print("done", w, sum(o is None for o in outs), "missing")
