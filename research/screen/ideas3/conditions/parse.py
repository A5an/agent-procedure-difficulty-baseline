"""Step 1: split every distinct policy text (415 units) into conditions and align each to a required step.
Segmentation uses the TEXT only (bullets, numbered steps, nested ANY-ONE blocks). Alignment to steps uses the rule tree
(steps.csv, top-level steps in order) and is used only to build training targets, never as a feature."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re, pandas as pd
S = ZR.SCREEN
R = json.load(open(f"{S}/ideas2/compile/results_A.json"))
st = {json.loads(l)["case_id"]: json.loads(l)["text"] for l in open(ZR.REPO + "/data/sopbench/statements.jsonl")}
steps = pd.read_csv(f"{S}/ideas2/steps/steps.csv", keep_default_na=False)
pol = lambda t: re.search(r"Policy for this action:\n(.*?)\nCustomer message:", t, re.S).group(1)
BUL = re.compile(r"^[•\-\*]\s+")
NUM = re.compile(r"^\d+\.\s+")

def blocks(lines):
    out = []
    for ln in lines:
        if not ln.strip():
            continue
        if BUL.match(ln):
            out.append([ln])
        elif out:
            out[-1].append(ln)
        else:
            out.append([ln])
    return out

def split_policy(p):
    lines = p.split("\n")
    if lines[0].startswith("ALL of these conditions must be met"):
        lines = lines[1:]
        conds = []
        for b in blocks(lines):
            first = BUL.sub("", b[0]).strip()
            if first.startswith("These steps must be completed in order"):
                for ln in b[1:]:
                    conds.append(dict(text=NUM.sub("", BUL.sub("", ln.strip())), is_or=0, n_alt=0, seq=1, nested=0))
            else:
                body = [BUL.sub("", b[0]).strip()] + [BUL.sub("", x.strip()) for x in b[1:]]
                is_or = int(first.startswith("ANY ONE"))
                nalt = 0
                if is_or:
                    nalt = sum(1 for x in b[1:] if BUL.match(x.strip()) and not x.strip().lstrip("•- ").startswith("ALL of these"))
                conds.append(dict(text=" ".join(body), is_or=is_or, n_alt=nalt, seq=0, nested=int(len(b) > 1)))
        return conds
    if lines[0].startswith("These steps must be completed in order"):
        return [dict(text=NUM.sub("", x.strip()), is_or=0, n_alt=0, seq=1, nested=0) for x in lines[1:] if x.strip()]
    return [dict(text=p.strip(), is_or=0, n_alt=0, seq=0, nested=0)]

if __name__ == "__main__":
    units = {}
    for r in R:
        units.setdefault(r["unit"], []).append(r["case_id"])
    rows, bad = [], []
    for u, cs in units.items():
        c0 = cs[0]
        conds = split_policy(pol(st[c0]))
        s = steps[steps.case_id == c0]
        top = s[(s.in_or == 0) & (s.step_type != "action")]
        if len(top) != len(conds):
            bad.append((u, len(conds), len(top))); continue
        for i, (cd, (_, sr)) in enumerate(zip(conds, top.iterrows())):
            rows.append(dict(unit=u, domain=c0.split("/")[0], procedure=c0.split("/")[1], idx=i, n_cond=len(conds), step_id=int(sr.step_id),
                             step_type=sr.step_type, step_name=sr["name"], step_kind=sr.kind, step_tools=sr.tools, **cd))
    print("bad", bad)
    df = pd.DataFrame(rows)
    df.to_csv("conds.csv", index=False)
    print(len(df), df.unit.nunique())
    json.dump(units, open("units.json", "w"))
