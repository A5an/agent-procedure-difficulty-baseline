"""Generic pre-mortem (FCg Sonnet, GG Gemini) and human-time (HT Gemini) on fresh benchmarks. usage: fcg.py sonnet|gemini"""
import sys, json, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "common")); sys.path.insert(0, str(HERE.parent / "fc"))
import vx, fc as FC1
ENV = {
 "tac": "A simulated software company. The agent works in a sandbox with a web browser, a terminal and Python, and must use the company's internal services: GitLab (code repositories), RocketChat (chat with simulated colleagues, who may have to be asked for information), ownCloud (files and documents) and Plane (project management). The task is graded by checkpoints on the final state of the services and files and on what the agent communicated.",
 "mcpmark": "The agent works through MCP tool servers for one service (Notion, GitHub, PostgreSQL, a local filesystem, or a Playwright-controlled web browser on mock websites). The task is graded by a script that checks the final state of the service.",
 "drafter": "The agent must write Python code that calls a provided library of about 46 functions to revise civil-engineering PDF drawings (add, modify, delete or map tables, text and vector entities). The task is graded by comparing the operations and arguments the code performs with the expected ones.",
}
P = """You are doing a pre-mortem. A tool-using LLM agent of the 2024-2025 generation (models such as GPT-4o, GPT-4.1, o3/o4-mini, Claude 3.5 to 4 Sonnet, Gemini 1.5 to 2.5, Llama 3 70B/405B, Qwen 2.5/3, DeepSeek V3, Kimi K2) will attempt the task below.

Environment and grading: {env}

Assume the agent WILL fail. Find the most likely concrete reasons, specific to THIS task, not generic ones.

=== TASK ===
{text}
=== END TASK ===

Return ONLY a JSON object, no prose before or after:
{{"failure_modes": [{{"what": "one sentence, specific to this task", "p": probability between 0 and 1 that a typical agent of that generation makes this mistake on this task}}],
  "p_success": probability between 0 and 1 that a typical agent of that generation fully succeeds on this task,
  "hardest_part": "one short phrase"}}
List at most 6 failure modes, most likely first."""
HT = ("Below is a task that an employee at a company would carry out with the company's internal tools.\n\n{text}\n\n"
      "Estimate how many minutes a trained, careful employee who knows the tools would need to complete this task correctly, "
      'from reading the request to the last action. Answer with one JSON object and nothing after it: {{"minutes": <number>}}')
items = json.load(open(HERE / "items.json"))
def prompt(it): return P.format(env=ENV[it["bench"]], text=it["text"])
if __name__ == "__main__":
    if sys.argv[1] == "sonnet":
        FC1.CACHE = HERE / "sonnet_cache.jsonl"; FC1.cache.clear(); FC1.WD = HERE / "work"
        if FC1.CACHE.exists():
            for l in FC1.CACHE.read_text().splitlines(): d = json.loads(l); FC1.cache[d["k"]] = d["r"]
        order = [i for i in items if i["bench"] != "drafter"] + [i for i in items if i["bench"] == "drafter"]
        with ThreadPoolExecutor(int(os.environ.get("NW", "8"))) as ex: outs = list(ex.map(lambda it: FC1.call(prompt(it)), order))
        json.dump({f'{it["bench"]}::{it["id"]}': o for it, o in zip(order, outs)}, open(HERE / "fcg_raw.json", "w"))
        print("done sonnet", sum(o is None for o in outs), "missing")
    else:
        C = vx.Client(HERE)
        jobs = [(prompt(it), "GG", True) for it in items] + [(HT.format(text=it["text"]), "HT", False) for it in items]
        with ThreadPoolExecutor(24) as ex: list(ex.map(lambda j: C.call(j[0], model="gemini-3.8-flash", tag=j[1], json_out=j[2], max_tokens=6000), jobs))
        out = {f'{it["bench"]}::{it["id"]}': {"GG": C.call(prompt(it), model="gemini-3.8-flash", tag="GG"),
                                               "HT": C.call(HT.format(text=it["text"]), model="gemini-3.8-flash", tag="HT", json_out=False)} for it in items}
        json.dump(out, open(HERE / "gem_raw.json", "w")); print("done gemini", C.ncalls, "fails", C.nfail)
