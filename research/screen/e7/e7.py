"""E7: zero-shot success forecast (direct question, Sonnet 5.5 via claude -p) on SWE-bench Verified, SWE-bench Pro, GSO, Terminal-Bench 2.0.
usage: e7.py test | run          (NW workers, default 10)"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, os, re, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "ideas10" / "fc"))
import fc as FC1

TXT = Path(ZR.DATA + "/e7_texts")
AP = Path(ZR.REPO + "/third_party/agent-psychometrics/data")
NAME = {"swebench_verified": "SWE-bench Verified", "swebench_pro": "SWE-bench Pro", "gso": "GSO", "terminalbench": "Terminal-Bench 2.0"}
ENV = {
 "swebench_verified": "The agent works in a container with a checkout of a popular open-source Python repository at the commit before the fix. It must edit the code so that the issue described below is resolved. It is graded by hidden tests: tests that failed before the fix must pass and the existing tests must keep passing.",
 "swebench_pro": "The agent works in a container with a checkout of a large open-source or commercial-grade repository (Python, Go, JavaScript or TypeScript) at the commit before the change. It must edit the code so that the issue or feature request below is implemented. It is graded by hidden tests: tests that failed before the change must pass and the existing tests must keep passing.",
 "gso": "The agent works in a container with a checkout of the repository and must make the named API or function faster without changing its behaviour. It is graded on performance tests: the task counts as solved only if the agent's patch keeps the code correct and reaches at least 95% of the speed-up achieved by the expert developers' own optimization commit.",
 "terminalbench": "The agent works in a sandboxed Linux container and completes the task by running shell commands (it may write and run code, install packages and inspect files). It is graded by automated tests on the final state of the container.",
}
PROMPT = """An LLM agent from the public leaderboard of {bench} will attempt the task below. The leaderboard mixes weaker and stronger systems; think of a typical submission.

Environment and grading: {env}

=== TASK ===
{text}
=== END TASK ===

Estimate the probability that a typical agent from that leaderboard fully succeeds on this task. Return ONLY a JSON object, no prose: {{"p_success": probability between 0 and 1}}"""
MAXC = 15000


def norm_pro(t):
    if t.startswith("instance_"): t = t[9:]
    return re.sub(r"-v(nan|[a-f0-9]+)$", "", t)


def items():
    out = []
    for b in ["swebench_verified", "swebench_pro", "gso"]:
        for l in open(TXT / f"{b}.jsonl"):
            r = json.loads(l)
            tid = norm_pro(r["instance_id"]) if b == "swebench_pro" else r["instance_id"]
            if b == "gso":
                text = f"Repository: {r['repo']}\nAPI or function to optimize: {r['api']}"
            else:
                text = f"Repository: {r['repo']}\n\nIssue:\n{r['problem_statement']}"
            out.append({"bench": b, "id": tid, "text": text[:MAXC]})
    for l in open(AP / "terminalbench" / "tasks.jsonl"):
        r = json.loads(l)
        out.append({"bench": "terminalbench", "id": r["task_id"], "text": r["problem_statement"][:MAXC]})
    return out


def prompt(it): return PROMPT.format(bench=NAME[it["bench"]], env=ENV[it["bench"]], text=it["text"])


if __name__ == "__main__":
    FC1.CACHE = HERE / "sonnet_cache.jsonl"; FC1.cache.clear(); FC1.WD = HERE / "work"; FC1.WD.mkdir(exist_ok=True)
    if FC1.CACHE.exists():
        for l in FC1.CACHE.read_text().splitlines(): d = json.loads(l); FC1.cache[d["k"]] = d["r"]
    its = items()
    if sys.argv[1] == "test":
        rnd = random.Random(0); its = [rnd.choice([i for i in its if i["bench"] == b]) for b in NAME]
    # smallest benchmarks first, so a stop on quota still leaves complete sets
    order = sorted(its, key=lambda i: {"terminalbench": 0, "gso": 1, "swebench_verified": 2, "swebench_pro": 3}[i["bench"]])
    with ThreadPoolExecutor(int(os.environ.get("NW", "10"))) as ex: outs = list(ex.map(lambda it: FC1.call(prompt(it)), order))
    res = {f'{it["bench"]}::{it["id"]}': o for it, o in zip(order, outs)}
    json.dump(res, open(HERE / ("test_raw.json" if sys.argv[1] == "test" else "e7_raw.json"), "w"))
    print("done", len(res), "missing", sum(v is None for v in res.values()), "cost_eq", round(FC1.stats["cost"], 2))
    if sys.argv[1] == "test":
        for k, v in res.items(): print(k, v)
