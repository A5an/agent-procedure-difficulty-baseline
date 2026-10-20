"""Small open model first-action scoring (L1). usage: llm_scores.py sop|tau2 [limit]
Qwen3-8B-Base (MLX 8 bit). Shared prefix per domain is KV cached; per case the suffix is fed, each candidate tool name is
scored by teacher forcing (cache trimmed back after each). Results appended to out/<bench>_raw.jsonl (resumable)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, re, time
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

S = Path(ZR.SCREEN)
sys.path.insert(0, str(S / "ideas"))
from cpulock import cpu_lock
HERE = S / "ideas7/small_model"; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
REPO = Path(ZR.REPO)
MODEL = ZR.DATA + "/models/qwen3-8b-base-mlx-8bit"
SOPENV = S / "ideas2/harness/sop"

PRINCIPLES = """### Core Operating Principles:
1. Action Selection: choose the most appropriate, direct, and best-fit action for the user's task or checking constraints. Avoid unnecessary function calls.
2. Action Validation: validate all required conditions in the specified order before proceeding with the target action. Use the most relevant tools to verify each prerequisite condition. Proceed with the target action only when all conditions are met. If any condition fails, explain why and decline the action.
3. Exit Conversation: exit the conversation if the request is completed or you cannot assist with this request."""
ASK = "The assistant decides what to do first and makes its first tool call.\nAssistant's first tool call (function name only):\n"


def required_tools(code, goal):
    pos = [m.start() for m in re.finditer(r"tools\.%s\b" % re.escape(goal), code)]
    cut = pos[0] if pos else len(code)
    names = []
    for m in re.finditer(r"tools\.(\w+)", code[:cut]):
        if m.group(1) not in names and m.group(1) != goal: names.append(m.group(1))
    return names


class Scorer:
    def __init__(self):
        self.model, self.tok = load(MODEL)
        self.model.eval()
    def enc(self, t): return self.tok.encode(t, add_special_tokens=False)
    def set_prefix(self, text):
        self.cache = make_prompt_cache(self.model)
        ids = self.enc(text); self.plen = len(ids)
        t0 = time.time()
        for i in range(0, len(ids), 1024):
            lg = self.model(mx.array([ids[i:i + 1024]]), cache=self.cache)
            mx.eval(lg)
        self.last = lg[0, -1].astype(mx.float32)
        self.last = self.last - mx.logsumexp(self.last)
        mx.eval(self.last)
        return time.time() - t0
    def run_case(self, segs, cands):
        """segs: list of (label, text) concatenated as suffix. Returns per-segment NLL arrays and candidate logprobs."""
        ids, spans = [], {}
        for lab, t in segs:
            e = self.enc(t); spans[lab] = (len(ids), len(ids) + len(e)); ids += e
        lg = self.model(mx.array([ids]), cache=self.cache)[0].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        tgt = mx.array(ids[1:])
        tok_lp = mx.take_along_axis(lp[:-1], tgt[:, None], axis=1)[:, 0]
        first = self.last[ids[0]]
        last_lp = lp[-1]
        mx.eval(tok_lp, first, last_lp)
        tok_nll = -np.concatenate([[float(first)], np.array(tok_lp)])
        seg_nll = {lab: tok_nll[a:b].tolist() for lab, (a, b) in spans.items()}
        res = {}
        for c in cands:
            cid = self.enc(c + "\n")
            s = float(last_lp[cid[0]])
            if len(cid) > 1:
                l2 = self.model(mx.array([cid[:-1]]), cache=self.cache)[0].astype(mx.float32)
                l2 = l2 - mx.logsumexp(l2, axis=-1, keepdims=True)
                v = mx.take_along_axis(l2, mx.array(cid[1:])[:, None], axis=1)
                mx.eval(v); s += float(v.sum())
                trim_prompt_cache(self.cache, len(cid) - 1)
            res[c] = s
        trim_prompt_cache(self.cache, len(ids))
        return seg_nll, res


def read_done(path):
    d = {}
    if path.exists():
        for l in path.read_text().splitlines():
            try: r = json.loads(l); d[r["case_id"]] = r
            except Exception: pass
    return d


def run_sop(limit=None):
    sys.path.insert(0, str(SOPENV)); os.chdir(SOPENV)
    from env.task import domain_assistant_keys
    stm = [json.loads(l) for l in open(REPO / "data/sopbench/statements.jsonl")]
    units = {r["case_id"]: r["unit"] for r in json.load(open(S / "ideas2/compile/results_A.json"))}
    progs = json.load(open(S / "ideas2/compile/programs_t0.json"))
    path = OUT / "sop_raw.jsonl"; done = read_done(path)
    sc = Scorer(); n = 0; t0 = time.time()
    for dom in ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]:
        cases = [s for s in stm if s["case_id"].startswith(dom + "/")]
        if limit: cases = cases[:limit]
        todo = [s for s in cases if s["case_id"] not in done]
        if not todo: continue
        a = domain_assistant_keys[dom]
        role = cases[0]["text"].split("\n")[0].split(". ", 1)[1]
        tl = []
        for x in a.actions:
            if x["name"] == "internal_get_database": continue
            props = list(x.get("parameters", {}).get("properties", {}))
            tl.append(f"- {x['name']}({', '.join(props)}): {a.action_descriptions[x['name']]} {a.action_returns[x['name']]}")
        prefix = f"Roleplay as an assistant that helps the user with his request. You and your functions are the only way the user can receive services.\n\n### Role Description:\n{role}\n\n{PRINCIPLES}\n\n### Available tools:\n" + "\n".join(tl) + "\n\n"
        dt = sc.set_prefix(prefix); print(dom, "prefix tokens", sc.plen, f"{dt:.1f}s", flush=True)
        for s in todo:
            cid = s["case_id"]; goal = cid.split("/")[1]
            m = re.search(r"Policy for this action:\n(.*)\nCustomer message: (.*)\nAvailable tools:", s["text"], re.S)
            pol, msg = m.group(1), m.group(2)
            req = required_tools(progs[units[cid]]["code"], goal)
            cands = [goal] + [r for r in req]
            segs = [("hdr", f"### Action constraints for the requested action\n* {goal}:\n"), ("policy", pol), ("mid", "\n\n### Conversation\nCustomer: "),
                    ("msg", msg), ("ask", "\n\n" + ASK)]
            nll, res = sc.run_case(segs, cands)
            # per-bullet NLL of the policy (gating rules)
            rec = {"case_id": cid, "goal": goal, "req": req, "lp": res, "nll_policy": nll["policy"], "nll_msg": nll["msg"], "policy": pol}
            with open(path, "a") as f: f.write(json.dumps(rec) + "\n")
            n += 1
            if n % 20 == 0: print(n, f"{time.time() - t0:.0f}s", flush=True)
    print("done", n, f"{time.time() - t0:.0f}s")


def run_tau2(limit=None):
    stm = [json.loads(l) for l in open(REPO / "data/tau2/statements.jsonl")]
    TT = json.load(open(S / "ideas3/plan_tools/tool_text.json"))
    TY = json.load(open(S / "ideas3/plan_tools/tool_types.json"))
    POL = Path(ZR.DATA + "/tau2-domains/tasks")
    path = OUT / "tau2_raw.jsonl"; done = read_done(path)
    sc = Scorer(); n = 0; t0 = time.time()
    for dom in ["airline", "retail", "telecom"]:
        cases = [s for s in stm if s["case_id"].startswith(dom + "/")]
        if limit: cases = cases[:limit]
        todo = [s for s in cases if s["case_id"] not in done]
        if not todo: continue
        pol = (POL / "telecom_main_policy.md").read_text() + "\n\n" + (POL / "telecom_tech_support_manual.md").read_text() if dom == "telecom" else (POL / f"{dom}_policy.md").read_text()
        names = [json.loads(l)["name"] for l in TT[dom].splitlines() if l.strip().startswith("{")]
        prefix = (f"You are a customer service agent. Follow the domain policy and use the tools.\n\n### Domain policy ({dom}):\n{pol}\n\n"
                  f"### Agent tools (one JSON per line):\n{TT[dom]}\n\n{PRINCIPLES}\n\n")
        dt = sc.set_prefix(prefix); print(dom, "prefix tokens", sc.plen, f"{dt:.1f}s", len(names), flush=True)
        for s in todo:
            cid = s["case_id"]
            segs = [("hdr", "### Conversation\n"), ("msg", s["text"]), ("mid", "\n\n"), ("ask", ASK)]
            nll, res = sc.run_case(segs, names)
            rec = {"case_id": cid, "lp": res, "types": {k: TY.get(f"{dom}/{k}", "?") for k in names}, "nll_msg": nll["msg"]}
            with open(path, "a") as f: f.write(json.dumps(rec) + "\n")
            n += 1
            if n % 20 == 0: print(n, f"{time.time() - t0:.0f}s", flush=True)
    print("done", n, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    which = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    with cpu_lock("small_model_" + which):
        if which in ("sop", "all"): run_sop(limit)
        if which in ("tau2", "all"): run_tau2(limit)
