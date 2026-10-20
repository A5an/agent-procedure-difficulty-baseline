"""E8 / E8-0 runner: single-model arms and the two-phase router (A until the first target-action proposal, then B).
Reuses the harness study code in ../ideas2/harness (Vertex client, SOPBench release, evaluator). See PREREG.md.

  .venv/bin/python run_e8.py --arm pro>flash            # arm names: see ARMS
  .venv/bin/python run_e8.py --arm pro --max_tokens 512 --carry_only --tag e80
"""
import os, sys, json, csv, copy, time, threading, argparse, io, contextlib, re
E8 = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(E8), "ideas2", "harness")
sys.path.insert(0, HARNESS)
import run as R                      # chdirs into the SOPBench copy, patches nothing for H0 runs
from swarm.core import Agent
from swarm import vclient
from swarm.vclient import VClient, ctx, BudgetExceeded

vclient.PRICES.update({
    "gemini-2.5-pro": (1.25, 10.00, 0.125),        # prompts up to 200k tokens
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01),
})
vclient.LEDGER = os.path.join(E8, "ledger.jsonl")

# a parallel flash run on 13 Oct hung with all workers waiting on open connections; use a shorter read timeout and retry
import requests as _rq
_orig_post = _rq.post
# gemini-2.5-* on the global endpoint returns 429 most of the time (13 Oct); spread those calls over regional endpoints
# that served each model without 429 in a probe at 21:00 (us-central1, us-east5 and later us-west1 were saturated)
_REG = {"gemini-2.5-flash-lite": ["europe-west4", "europe-west1", "europe-north1"],
        "gemini-2.5-flash": ["europe-west4", "europe-west1", "europe-north1", "europe-west3", "europe-west8", "europe-west9",
                             "europe-southwest1", "europe-central2", "northamerica-northeast1", "us-south1", "us-west4", "asia-northeast1"],
        "gemini-2.5-pro": ["europe-west4", "europe-west1", "europe-north1", "europe-west8", "europe-southwest1", "europe-central2",
                           "northamerica-northeast1", "us-south1", "us-west4", "asia-northeast1", "us-west1", "us-central1", "us-east1"]}
_rr = {"i": 0}
def _post_timeout(url, *a, **k):
    k["timeout"] = (20, 150)
    m = re.search(r"/locations/global/publishers/google/models/(gemini-2\.5-[a-z-]+):", url)
    if m and m.group(1) in _REG:
        regs = _REG[m.group(1)]
        with vclient._lock:
            reg = regs[_rr["i"] % len(regs)]; _rr["i"] += 1
        url = url.replace("https://aiplatform.googleapis.com", f"https://{reg}-aiplatform.googleapis.com").replace("/locations/global/", f"/locations/{reg}/")
    return _orig_post(url, *a, **k)
vclient.requests.post = _post_timeout
import urllib3.util.connection as _uc
_uc.HAS_IPV6 = False          # the hung connections were all IPv6; force IPv4
OUT = os.path.join(E8, "episodes"); os.makedirs(OUT, exist_ok=True)

M = {"flash": "gemini-2.5-flash", "pro": "gemini-2.5-pro", "lite": "gemini-2.5-flash-lite", "g38": "gemini-3.8-flash"}
ARMS = {"flash": ("flash", None), "pro": ("pro", None), "pro>flash": ("pro", "flash"), "flash>pro": ("flash", "pro"),
        "lite": ("lite", None), "g38": ("g38", None), "lite>g38": ("lite", "g38")}


class RouterClient(VClient):
    """A answers until it first proposes the target action; that proposal is dropped and B answers from then on."""
    def __init__(self, a, b, goal):
        self.A, self.B, self.goal = VClient(a), VClient(b), goal
        self.model_name = self.model_name_huggingface = f"{a}>{b}"
        self.kind = "gemini"; self.phase = "A"; self.log = []; self.n_a = 0; self.n_b = 0

    def _handoff_sigs(self):
        # A's calls stay in the history as plain function calls; Gemini 3 needs the documented dummy signature
        dummy = self.B.model_name.startswith("gemini-3")
        sigs = getattr(ctx, "sigs", {})
        for cid, part in list(sigs.items()):
            fc = part.get("functionCall")
            sigs[cid] = {"functionCall": fc, "thoughtSignature": "skip_thought_signature_validator"} if dummy else {"functionCall": fc}
        ctx.sigs = sigs

    @staticmethod
    def _summ(comp):
        msg = comp.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            return {"call": tc.function.name, "args": tc.function.arguments}
        return {"call": None, "text": (msg.content or "")[:300]}

    def inference(self, p, debug=False, mode="chat", tool_call_mode="fc"):
        if self.phase == "A":
            out = self.A.inference(p, debug, mode, tool_call_mode); self.n_a += 1; ctx.usage[-1]["model"] = self.A.model_name
            s = self._summ(out["completion"])
            if s["call"] != self.goal:
                return out
            self.phase = "B"; self._handoff_sigs()
            outb = self.B.inference(p, debug, mode, tool_call_mode); self.n_b += 1; ctx.usage[-1]["model"] = self.B.model_name
            self.log.append({"event": "handoff", "after_a_calls": self.n_a, "a_proposal": s, "b_first": self._summ(outb["completion"])})
            return outb
        self.n_b += 1
        out = self.B.inference(p, debug, mode, tool_call_mode); ctx.usage[-1]["model"] = self.B.model_name
        return out


def run_episode(arm, case_id, task, max_tokens, temp):
    a, b = ARMS[arm]
    args = R.make_args(task["domain"], "none")
    dep_in, dep_full, dep_descr = R.deps(task["domain"])
    client = RouterClient(M[a], M[b], task["user_goal"]) if b else VClient(M[a])
    ag = Agent(name="domain assistant", client=client, temperature=temp, top_p=0.01, max_tokens=max_tokens,
               tool_call_mode="fc", instructions="", functions=[])
    user = Agent(name="user", client=None, default_response="", response_repeat=True)
    ctx.tag = (client.model_name, arm, case_id); ctx.usage = []; ctx.sigs = {}; ctx.guard = None
    with contextlib.redirect_stdout(io.StringIO()):
        res = R.rs.run_task_simulation(args, copy.deepcopy(task), dep_in, dep_full, dep_descr, ag, user)
    router = {"log": client.log, "n_a": client.n_a, "n_b": client.n_b} if b else None
    return res, router


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--cases", default=os.path.join(HARNESS, "sample.csv"))
    ap.add_argument("--carry_only", action="store_true"); ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--total_cap", type=float, default=80.0); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    vclient.CAPS["total"] = args.total_cap
    vclient.load_ledger()
    rows = list(csv.DictReader(open(args.cases)))
    if args.carry_only: rows = [r for r in rows if r["carry"] == "True"]
    ids = [r["case_id"] for r in rows][: args.limit]
    tasks = R.load_tasks()
    path = os.path.join(OUT, f"{args.arm.replace('>', '_to_')}{'__' + args.tag if args.tag else ''}.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if not r.get("error"): done.add(r["case_id"])
    todo = [c for c in ids if c not in done]
    print(f"{args.arm} {args.tag}: {len(todo)} to run, {len(done)} done; spent {vclient.spent():.2f}", flush=True)
    wlock = threading.Lock(); stop = {"v": False}

    def work(cid):
        if stop["v"]: return
        t0 = time.time(); err = None; res = None; router = None; temp_used = 0.0
        for attempt in range(3):
            temp_used = 0.0   # 13 Oct: retries were firing on infrastructure errors (404, 429, timeouts), so they stay at temperature 0
            try:
                res, router = run_episode(args.arm, cid, tasks[cid], args.max_tokens, temp_used); err = None; break
            except BudgetExceeded as e:
                stop["v"] = True; err = "BUDGET " + str(e); break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:300]}"
        rec = {"case_id": cid, "arm": args.arm, "tag": args.tag, "error": err, "secs": round(time.time() - t0, 1),
               "temp": temp_used, "usage": getattr(ctx, "usage", []), "max_tokens": args.max_tokens, "router": router}
        if res:
            rec["interaction"] = res["interaction"]; rec["database"] = res["database"]
            rec["usd"] = sum(u["usd"] for u in rec["usage"])
        with wlock:
            with open(path, "a") as f: f.write(json.dumps(rec, default=str) + "\n")
            print(f"{cid} {'ERR ' + err if err else 'ok'} {rec['secs']}s spent={vclient.spent():.2f}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(args.workers) as ex: list(ex.map(work, todo))
    print("finished", args.arm, args.tag, "spent", round(vclient.spent(), 2), flush=True)


if __name__ == "__main__":
    main()
