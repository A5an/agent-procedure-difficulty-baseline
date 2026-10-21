"""Round 9 probe: pre-generation hidden-state extraction (L1, local white box). usage: extract.py [limit_per_group]
Qwen3-8B-Base MLX 8 bit. Per domain the policy/tool prefix is KV cached. Per case, chunk A = case policy header
(SOPBench) or conversation header, chunk B = request + fixed cue. Saves states/<bench>__<case>.npz (float16), resumable.
Saved: last[L,4096] state at last cue token, mean_req[L,4096] mean over request tokens, ref[L,4096] state at the last
token of chunk A (policy-only reference), plus scalars: nll_cond (request NLL with policy), nll_unc (without), token counts.
No outcomes are read. No cpu_lock is held (GPU only)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json, re, time
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.models.base import create_attention_mask

S = Path(ZR.SCREEN)
HERE = S / "ideas9/probe"; OUT = HERE / "states"
REPO = Path(ZR.REPO)
DATA = Path(ZR.DATA)
MODEL = ZR.DATA + "/models/qwen3-8b-base-mlx-8bit"
SOPENV = S / "ideas2/harness/sop"
LAYERS = [12, 16, 20, 24, 28, 32]   # hidden state after this many transformer blocks (1-indexed count)
PRINCIPLES = """### Core Operating Principles:
1. Action Selection: choose the most appropriate, direct, and best-fit action for the user's task or checking constraints. Avoid unnecessary function calls.
2. Action Validation: validate all required conditions in the specified order before proceeding with the target action. Use the most relevant tools to verify each prerequisite condition. Proceed with the target action only when all conditions are met. If any condition fails, explain why and decline the action.
3. Exit Conversation: exit the conversation if the request is completed or you cannot assist with this request."""
CUE = "The assistant decides what to do first.\nAssistant's first step:"


class P:
    def __init__(self):
        self.model, self.tok = load(MODEL); self.model.eval()
        self.layers = self.model.model.layers
    def enc(self, t): return self.tok.encode(t, add_special_tokens=False)
    def fwd(self, ids, cache):
        """returns logits[T,V] float32 log-softmax-ready, and dict L -> hidden[T,4096] (float32)"""
        m = self.model.model
        h = m.embed_tokens(mx.array([ids]))
        mask = create_attention_mask(h, cache[0] if cache else None)
        hs = {}
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, cache[i] if cache else None)
            if (i + 1) in LAYERS: hs[i + 1] = h[0]
        out = m.norm(h)
        lg = self.model.model.embed_tokens.as_linear(out) if self.model.args.tie_word_embeddings else self.model.lm_head(out)
        return lg[0].astype(mx.float32), hs
    def set_prefix(self, text):
        self.cache = make_prompt_cache(self.model); ids = self.enc(text); self.plen = len(ids)
        for i in range(0, len(ids), 1024):
            lg = self.model(mx.array([ids[i:i + 1024]]), cache=self.cache); mx.eval(lg)
        self.last = lg[0, -1].astype(mx.float32); self.last = self.last - mx.logsumexp(self.last); mx.eval(self.last)
    def case(self, A, msg, cue):
        a, b, c = self.enc(A), self.enc(msg), self.enc(cue)
        lgA, hA = self.fwd(a, self.cache)
        lastA = lgA[-1] - mx.logsumexp(lgA[-1])
        lgB, hB = self.fwd(b + c, self.cache)
        lp = lgB - mx.logsumexp(lgB, axis=-1, keepdims=True)
        tgt = mx.array((b + c)[1:])
        tl = mx.take_along_axis(lp[:-1], tgt[:, None], axis=1)[:, 0]
        first = lastA[b[0]]
        nb = len(b)
        nll_tok = -mx.concatenate([first[None], tl[:nb - 1]])
        ref = mx.stack([hA[L][-1] for L in LAYERS]).astype(mx.float32); last = mx.stack([hB[L][-1] for L in LAYERS]).astype(mx.float32)
        mreq = mx.stack([hB[L][:nb].astype(mx.float32).mean(0) for L in LAYERS])
        mx.eval(nll_tok, ref, last, mreq)
        trim_prompt_cache(self.cache, len(a) + len(b) + len(c))
        # no-policy pass: request given only the bare conversation header
        return dict(ref=np.array(ref), last=np.array(last), mreq=np.array(mreq), nll_cond=float(nll_tok.mean()),
                    nll_tok=np.array(nll_tok), n_req=nb, n_A=len(a), n_pre=self.plen), (b,)
    def nll_nopolicy(self, hdr, msg_ids):
        a = self.enc(hdr); ids = a + msg_ids
        lg, _ = self.fwd(ids, None)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        tl = mx.take_along_axis(lp[len(a) - 1:-1], mx.array(msg_ids)[:, None], axis=1)[:, 0]
        mx.eval(tl); return -np.array(tl)


def save(bench, cid, r, nll_unc):
    mx.clear_cache()
    f = OUT / (bench + "__" + cid.replace("/", "__") + ".npz")
    np.savez(f, ref=r["ref"].astype(np.float16), last=r["last"].astype(np.float16), mreq=r["mreq"].astype(np.float16),
             nll_tok=r["nll_tok"], nll_unc_tok=nll_unc,
             sc=np.array([r["nll_cond"], float(nll_unc.mean()), r["n_req"], r["n_A"], r["n_pre"]]))


def todo(bench, cases):
    return [s for s in cases if not (OUT / (bench + "__" + s["case_id"].replace("/", "__") + ".npz")).exists()]


def run(limit=None):
    mx.set_cache_limit(1 << 30)
    p = P(); n = 0; t0 = time.time()
    # banking
    stm = [json.loads(l) for l in open(REPO / "data/tauk_banking/statements.jsonl")]
    if limit: stm = stm[:limit]
    td = todo("tauk_banking", stm)
    if td:
        B = DATA / "tau2-banking"; pr = B / "prompts"
        comps = "\n\n".join((pr / "components" / f).read_text() for f in ["policy_header.md", "shell_instructions.md", "additional_instructions.md"])
        stems = sorted({re.sub(r"_\d+\.json$", "", f.name)[4:] for f in (B / "documents").iterdir()})
        tools = "- shell(command): run a Unix command in the knowledge base directory\n- get_current_time(): current time\n- transfer_to_human_agents(summary): transfer to a human agent"
        prefix = (f"You are a customer service agent. Follow the policy and use the tools.\n\n{comps}\n\n### Knowledge base topics (document groups):\n" + "\n".join(stems) +
                  f"\n\n### Agent tools:\n{tools}\n\n{PRINCIPLES}\n\n")
        p.set_prefix(prefix); print("banking prefix tokens", p.plen, flush=True)
        for s in td:
            req = s["text"].split("Customer scenario:", 1)[-1].strip()
            A = "### Conversation\nCustomer scenario:\n"
            r, (b,) = p.case(A, req, "\n\n" + CUE)
            save("tauk_banking", s["case_id"], r, p.nll_nopolicy(A, b)); n += 1
            if n % 10 == 0: print(n, f"{time.time() - t0:.0f}s", flush=True)
    # tau2
    stm = [json.loads(l) for l in open(REPO / "data/tau2/statements.jsonl")]
    TT = json.load(open(S / "ideas3/plan_tools/tool_text.json")); POL = DATA / "tau2-domains/tasks"
    for dom in ["airline", "retail", "telecom"]:
        cases = [s for s in stm if s["case_id"].startswith(dom + "/")]
        if limit: cases = cases[:limit]
        td = todo("tau2", cases)
        if not td: continue
        pol = ((POL / "telecom_main_policy.md").read_text() + "\n\n" + (POL / "telecom_tech_support_manual.md").read_text()) if dom == "telecom" else (POL / f"{dom}_policy.md").read_text()
        prefix = (f"You are a customer service agent. Follow the domain policy and use the tools.\n\n### Domain policy ({dom}):\n{pol}\n\n"
                  f"### Agent tools (one JSON per line):\n{TT[dom]}\n\n{PRINCIPLES}\n\n")
        p.set_prefix(prefix); print(dom, "prefix tokens", p.plen, flush=True)
        for s in td:
            req = s["text"].split("Customer scenario:", 1)[-1].strip()
            A = "### Conversation\nCustomer scenario:\n"
            r, (b,) = p.case(A, req, "\n\n" + CUE)
            save("tau2", s["case_id"], r, p.nll_nopolicy(A, b)); n += 1
            if n % 10 == 0: print(n, f"{time.time() - t0:.0f}s", flush=True)
    # sopbench
    sys.path.insert(0, str(SOPENV)); os.chdir(SOPENV)
    from env.task import domain_assistant_keys
    stm = [json.loads(l) for l in open(REPO / "data/sopbench/statements.jsonl")]
    for dom in ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]:
        cases = [s for s in stm if s["case_id"].startswith(dom + "/")]
        if limit: cases = cases[:limit]
        td = todo("sopbench", cases)
        if not td: continue
        a = domain_assistant_keys[dom]; role = cases[0]["text"].split("\n")[0].split(". ", 1)[1]
        tl = []
        for x in a.actions:
            if x["name"] == "internal_get_database": continue
            props = list(x.get("parameters", {}).get("properties", {}))
            tl.append(f"- {x['name']}({', '.join(props)}): {a.action_descriptions[x['name']]} {a.action_returns[x['name']]}")
        prefix = f"Roleplay as an assistant that helps the user with his request. You and your functions are the only way the user can receive services.\n\n### Role Description:\n{role}\n\n{PRINCIPLES}\n\n### Available tools:\n" + "\n".join(tl) + "\n\n"
        p.set_prefix(prefix); print(dom, "prefix tokens", p.plen, flush=True)
        for s in td:
            goal = s["case_id"].split("/")[1]
            m = re.search(r"Policy for this action:\n(.*)\nCustomer message: (.*)\nAvailable tools:", s["text"], re.S)
            pol, msg = m.group(1), m.group(2)
            A = f"### Action constraints for the requested action\n* {goal}:\n{pol}\n\n### Conversation\nCustomer: "
            r, (b,) = p.case(A, msg, "\n\n" + CUE)
            save("sopbench", s["case_id"], r, p.nll_nopolicy("### Conversation\nCustomer: ", b)); n += 1
            if n % 10 == 0: print(n, f"{time.time() - t0:.0f}s", flush=True)
    print("done", n, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else None)
