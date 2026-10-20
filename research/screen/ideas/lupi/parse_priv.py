"""Part (a): privileged signals per case from raw trajectories. Writes privileged_sopbench.csv and privileged_tau2.csv."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, os, re, gc
sys.path.insert(0, ZR.SCREEN + "/ideas")
from cpulock import cpu_lock
import numpy as np, pandas as pd
SOP = ZR.DATA + "/SOPBench-d2622008"
TAU = ZR.DATA + "/tau2-domains"
REPO = ZR.REPO
HERE = os.path.dirname(os.path.abspath(__file__))
DOM = ["bank","dmv","healthcare","hotel","library","online_market","university"]

def sop():
    agents = [json.loads(l)["subject_id"] for l in open(f"{REPO}/data/sopbench/responses.jsonl")]
    fl = pd.read_csv(ZR.SCREEN + "/pertype/flags.csv.gz") \
        if os.path.exists(ZR.SCREEN + "/pertype/flags.csv.gz") else None
    rows = []
    for d in DOM:
        td = json.load(open(f"{SOP}/data/{d}_tasks.json"))
        ids = [f"{d}/{g}/{j}" for g, l in td.items() for j, _ in enumerate(l)]
        for a in agents:
            recs = json.load(open(f"{SOP}/output/{d}/ast_{a}-dep_full-fmt_structured-tool_full-shuffle_False.json"))
            assert len(recs) == len(ids)
            for cid, r in zip(ids, recs):
                if not r["interactions"]: continue
                goal = r["task"].get("user_goal"); na = nt = 0; hit = 0
                for m in r["interactions"][0]["interaction"]:
                    if not isinstance(m, dict) or "tool_call_id" in m or m.get("sender") == "user": continue
                    na += 1
                    for tc in (m.get("tool_calls") or []):
                        nm = tc["function"]["name"]
                        if nm.lower() in ("n/a","na","none","null"): continue
                        nt += 1; hit |= (nm == goal)
                ev = (r.get("evaluations") or [{}])[0]
                rows.append(dict(agent=a, case_id=cid, n_tool=nt, n_asst=na, called_target=hit, zero_tool=int(nt == 0),
                                 s_dirgraph=ev.get("dirgraph_satisfied"), s_acc=ev.get("action_called_correctly")))
            del recs; gc.collect()
    df = pd.DataFrame(rows)
    if fl is not None:
        fl = fl[["agent","case_id","dirgraph_satisfied","action_called_correctly"]]
        df = df.merge(fl, on=["agent","case_id"], how="left")
    else:
        df["dirgraph_satisfied"], df["action_called_correctly"] = df.s_dirgraph, df.s_acc
    for c in ["dirgraph_satisfied","action_called_correctly"]:
        df[c] = df[c].where(df[c].notna(), df["s_"+("dirgraph" if "dir" in c else "acc")]).astype(float)
    g = df.groupby("case_id")
    out = pd.DataFrame({"mean_tool_calls": g.n_tool.mean(), "mean_asst_msgs": g.n_asst.mean(),
        "share_called_target": g.called_target.mean(), "share_zero_tool": g.zero_tool.mean(),
        "share_dirgraph": g.dirgraph_satisfied.mean(), "share_action_correct": g.action_called_correctly.mean(),
        "n_runs": g.size()})
    out.reset_index().to_csv(f"{HERE}/privileged_sopbench.csv", index=False)
    print(out.describe().T)

def tau():
    ids = json.load(open(f"{REPO}/data/tau2/id_map.json"))
    subs = [json.loads(l)["subject_id"] for l in open(f"{REPO}/data/tau2/responses.jsonl")]
    files = [f for f in sorted(os.listdir(f"{TAU}/trajectories")) if f.split("__")[0] in subs]
    rows = []
    for f in files:
        dom = re.search(r"airline|retail|telecom", f.split("__",1)[1]).group(0)
        d = json.load(open(f"{TAU}/trajectories/{f}"))
        for s in d["simulations"]:
            key = f"{dom}:{s['task_id']}"
            if key not in ids: continue
            msgs = s["messages"]; nt = 0; tr = 0
            for m in msgs:
                if m["role"] == "assistant":
                    for c in (m.get("tool_calls") or []):
                        nt += 1; tr |= ("transfer" in c["name"])
            rows.append(dict(sub=f.split("__")[0], case_id=ids[key], n_msgs=len(msgs), n_tool=nt, transfer=tr, term=s["termination_reason"]))
        del d; gc.collect()
    df = pd.DataFrame(rows)
    print(len(files), "files", df.term.value_counts().to_dict())
    g = df.groupby("case_id")
    out = pd.DataFrame({"mean_msgs": g.n_msgs.mean(), "mean_tool_calls": g.n_tool.mean(), "share_transfer": g.transfer.mean(), "n_runs": g.size()})
    for t in df.term.unique():
        out["term_" + t] = g.term.apply(lambda x, t=t: (x == t).mean())
    out.reset_index().to_csv(f"{HERE}/privileged_tau2.csv", index=False)
    print(out.describe().T)

if __name__ == "__main__":
    with cpu_lock("lupi"):
        if not os.path.exists(f"{HERE}/privileged_sopbench.csv"): sop()
        if not os.path.exists(f"{HERE}/privileged_tau2.csv"): tau()
