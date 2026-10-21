"""Step 1: required steps per SOPBench case, from the rule tree (constraints_original) and the directed action graph.
Writes steps.csv (one row per case and step) and step_type_mapping.csv (constraint name -> type, tool kind, counts).
Labels derived here are FROM THE RULE TREE (restates the SOP text). Truth values come from the case database and are
used only as evaluation targets, never as features."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import copy, json, sys, csv, collections
ROOT = ZR.DATA + "/SOPBench-d2622008"
sys.path.insert(0, ROOT)
from env.variables import domain_assistant_keys, domain_keys
import pandas as pd
from types import SimpleNamespace
sys.path.insert(0, ZR.SCREEN + "/ideas2/steps")
from types_map import step_type  # noqa

DOMS = ["bank", "dmv", "healthcare", "hotel", "library", "online_market", "university"]


def leaf_tools(cp):
    if not cp:
        return []
    if cp[0] == "single":
        return [cp[1]]
    out = []
    for d in cp[1]:
        out += leaf_tools(d)
    return list(dict.fromkeys(out))


def cp_tree(cp):
    if not cp:
        return None
    if cp[0] == "single":
        return ["single", cp[1]]
    return [cp[0] if cp[0] in ("and", "or") else "and", [cp_tree(d) for d in cp[1]]]


def tool_tree(name, links, cp):
    n = name.replace("not ", "")
    if n in links:
        return ["single", links[n][0]]
    if n.startswith("internal_"):
        return ["or", [["single", n], ["single", "internal_get_database"]]]
    return cp_tree(cp.get(n))


def tool_kind(name, links, cp):
    n = name.replace("not ", "")
    if n in links:
        return "link", [links[n][0]]
    if n.startswith("internal_"):
        return "check", [n, "internal_get_database"]
    t = leaf_tools(cp.get(n))
    if t:
        return "fetch", t
    return "none", []


def flatten(tree, in_or=False, out=None, or_id=[0]):
    """Return nested structure: ('leaf', leaf) | ('and'/'or', [children])"""
    if tree[0] == "single":
        return ("leaf", tree)
    return (tree[0] if tree[0] in ("and", "or") else "and", [flatten(c) for c in tree[1]])


def truth(node, tv):
    if node[0] == "leaf":
        return tv[id(node[1])]
    vals = [truth(c, tv) for c in node[1]]
    if any(v is None for v in vals):
        return None
    return all(vals) if node[0] == "and" else any(vals)


rows, dbg = [], collections.Counter()
TREE_OK = {}
for dom in DOMS:
    A = domain_assistant_keys[dom]
    links, cp = A.constraint_links, A.constraint_processes
    d = json.load(open(f"{ROOT}/data/{dom}_tasks.json"))
    for proc, cases in d.items():
        for idx, c in enumerate(cases):
            cid = f"{dom}/{proc}/{idx}"
            tree = c["constraints_original"]
            uk = c["user_known"]
            ds = domain_keys[dom](copy.deepcopy(c["initial_database"]), A.action_innate_dependencies, c["constraint_parameters"])
            goal = proc
            gnode = c["directed_action_graph"]["nodes"][0]
            action_params = gnode[1]
            if not tree:
                rows.append(dict(case_id=cid, step_id=0, step_type="action", name=goal, negated=0, tools="", kind="action",
                                 truth=1, in_or=0, group=-1, params=json.dumps(action_params), should_succeed=int(c["action_should_succeed"]), n_leaves=0))
                dbg["no_tree"] += 1
                continue
            node = flatten(tree)
            tv = {}
            def ev(l):
                nm = l[1].replace("not ", "")
                neg = l[1].startswith("not ")
                try:
                    if nm in links:
                        act = links[nm][0]
                        gnodes = [n for n in c["directed_action_graph"]["nodes"] if not isinstance(n, str) and n[0] == act]
                        mp = gnodes[0][1] if gnodes else {}
                        args = {k: uk.get(v) for k, v in mp.items()}
                        if not gnodes:
                            args = {k: uk[k] for k in uk if k in ("username", "identification", "admin_password", "password")}
                        r = getattr(ds, act)(**args)
                        v = bool(r) != neg
                    else:
                        v = bool(ds.domain_dep._process(l, **uk))
                    return v
                except Exception as e:
                    dbg["eval_err_" + nm] += 1
                    return neg   # underlying lookup failed (entity missing): the fact is False
            def walk(n):
                if n[0] == "leaf":
                    tv[id(n[1])] = ev(n[1])
                else:
                    for x in n[1]:
                        walk(x)
            walk(node)
            ts = truth(node, tv)
            TREE_OK[cid] = int(ts == bool(c["action_should_succeed"]))
            dbg["tree_truth_match" if ts == bool(c["action_should_succeed"]) else ("tree_truth_none" if ts is None else "tree_truth_MISMATCH")] += 1
            # steps: top-level leaves outside any OR are steps; each topmost OR is one or_group step
            steps = []
            def collect(n, under_or=False):
                if n[0] == "leaf":
                    steps.append(("leaf", n)); return
                if n[0] == "or":
                    steps.append(("or", n)); return
                for x in n[1]:
                    collect(x)
            collect(node)
            def leaves(n):
                return [n[1]] if n[0] == "leaf" else [l for x in n[1] for l in leaves(x)]
            sid = 0
            for kind, n in steps:
                if kind == "leaf":
                    l = n[1]; nm = l[1].replace("not ", ""); tk, tools = tool_kind(l[1], links, cp)
                    rows.append(dict(case_id=cid, step_id=sid, step_type=step_type(l[1]), name=nm, negated=int(l[1].startswith("not ")),
                                     tools="|".join(tools), tree=json.dumps(tool_tree(l[1], links, cp)), kind=tk, truth=(None if tv[id(l)] is None else int(tv[id(l)])), in_or=0, group=-1,
                                     params=json.dumps(l[2] or {}), should_succeed=int(c["action_should_succeed"])))
                    sid += 1
                else:
                    ch = leaves(n)
                    rows.append(dict(case_id=cid, step_id=sid, step_type="or_group", name="OR(" + ",".join(x[1].replace("not ", "") for x in ch) + ")",
                                     negated=0, tools="", kind="or", truth=(None if truth(n, tv) is None else int(truth(n, tv))), in_or=0, group=sid,
                                     params="", should_succeed=int(c["action_should_succeed"])))
                    gid = sid; sid += 1
                    for l in ch:
                        nm = l[1].replace("not ", ""); tk, tools = tool_kind(l[1], links, cp)
                        rows.append(dict(case_id=cid, step_id=sid, step_type=step_type(l[1]), name=nm, negated=int(l[1].startswith("not ")),
                                         tools="|".join(tools), tree=json.dumps(tool_tree(l[1], links, cp)), kind=tk, truth=(None if tv[id(l)] is None else int(tv[id(l)])), in_or=1, group=gid,
                                         params=json.dumps(l[2] or {}), should_succeed=int(c["action_should_succeed"])))
                        sid += 1
            rows.append(dict(case_id=cid, step_id=sid, step_type="action", name=goal, negated=0, tools=goal, kind="action", truth=int(c["action_should_succeed"]),
                             in_or=0, group=-1, params=json.dumps(action_params), should_succeed=int(c["action_should_succeed"])))
df = pd.DataFrame(rows)
df["tree_ok"] = df.case_id.map(TREE_OK).fillna(1).astype(int)
df.to_csv("steps.csv", index=False)
print(dict(dbg))
print(df.case_id.nunique(), len(df))
m = df[df.step_type != "action"].groupby(["name", "step_type", "kind"]).agg(n=("case_id", "size"), tools=("tools", "first")).reset_index()
m.to_csv("step_type_mapping.csv", index=False)
print(m.groupby("step_type").n.sum())
