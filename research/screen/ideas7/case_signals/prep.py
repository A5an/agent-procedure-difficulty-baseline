"""Case parsing for case_signals: message, action, conditions, tool args. Statement text only."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, re, json, ast, glob
S = ZR.SCREEN
sys.path.insert(0, S + "/ideas3/conditions")
import pandas as pd
import parse as PA   # split_policy
REPO = ZR.REPO
ENV = ZR.DATA + "/SOPBench-d2622008/env/domains"
tasks = pd.read_csv(f"{REPO}/data/sopbench/tasks.csv")
ST = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(f"{REPO}/data/sopbench/statements.jsonl")}
def tool_args():
    out = {}
    for f in glob.glob(f"{ENV}/*/*.py"):
        dom = f.split("/")[-2]
        for n in ast.walk(ast.parse(open(f).read())):
            if isinstance(n, ast.FunctionDef) and len(n.args.args) > 1 and not n.name.startswith("_"):
                a = [(x.arg, ast.unparse(x.annotation) if x.annotation else "") for x in n.args.args[1:]]
                if (dom, n.name) not in out or len(a) > len(out[(dom, n.name)]): out[(dom, n.name)] = a
    return out
TA = tool_args()
def parse_case(c):
    t = ST[c]
    pol = re.search(r"Policy for this action:\n(.*?)\nCustomer message:", t, re.S).group(1)
    msg = re.search(r"Customer message: (.*?)\nAvailable tools:", t, re.S).group(1).strip()
    act = re.search(r"Requested action: (\S+)", t).group(1)
    tools = re.search(r"Available tools: (.*)$", t, re.S).group(1).split(", ")
    return pol, msg, act, [x.strip() for x in tools]
def clean(x): return re.sub(r"\*\*", "", x).strip()
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
def sents(m): return [s for s in SENT.split(m) if len(s) > 3]
CASES = {}
for r in tasks.itertuples():
    pol, msg, act, tools = parse_case(r.case_id)
    conds = [dict(d, text=clean(d["text"])) for d in PA.split_policy(pol)]
    CASES[r.case_id] = dict(domain=r.domain, proc=r.procedure, pid=r.procedure_id, pol=pol, msg=msg, act=act, tools=tools, conds=conds, sents=sents(msg))
if __name__ == "__main__":
    n = sum(len(v["conds"]) * (len(v["sents"]) + 1) for v in CASES.values())
    print(len(CASES), "pairs", n, "avg conds", sum(len(v["conds"]) for v in CASES.values()) / len(CASES))
    c = "bank/apply_credit_card/0"; print(CASES[c]["sents"], CASES[c]["conds"], TA[("bank", "apply_credit_card")])
    miss = [v["act"] for v in CASES.values() if (v["domain"], v["act"]) not in TA]; print("missing tool args", set(miss))
