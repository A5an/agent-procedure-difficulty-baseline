"""Tidy table of every recorded SOPBench outcome we have, for the multi-fidelity model. No API calls.

Rows: agent, fid (real | synth | public), case_id, proc, domain, perform (1 = the action should be performed), fail.
  real   : our harness, 830 SOPBench cases (gemini-2.5-flash-lite, gemini-2.5-flash, gemini-3.8-flash, Sonnet 5.5)
  synth  : the same four agents on the 1,365 LLM-written probe cases (ideas2/emulate_run, ideas5)
  public : the 28 public SOPBench configs (release logs, other harness), used only as side information
The two bank procedures whose target tool is missing (cancel_credit_card, pay_bill_with_credit_card) are dropped.
"""
import os, json, csv
import pandas as pd

S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(S, "..", "baseline", "data", "sopbench")
BROKEN = {"bank/cancel_credit_card", "bank/pay_bill_with_credit_card"}


def proc_of(cid):
    return "/".join(cid.split("/")[:2])


def build():
    rows = []
    tasks = {r["case_id"]: r for r in csv.DictReader(open(os.path.join(BASE, "tasks.csv")))}
    perform = {c: int(r["should_succeed"]) for c, r in tasks.items()}
    for agent, run in (("lite", "lite"), ("flash", "flash")):
        for fid in ("real", "synth"):
            for r in json.load(open(os.path.join(S, "ideas2", "emulate_run", "runs", f"{fid}_{run}", "results.json"))):
                if r["success"] is None: continue
                cid = r["cid"]
                rows.append((agent, fid, cid, f"{r['domain']}/{r['proc']}", r["domain"], int(r["perform"]), 1 - int(r["success"])))
    for agent, d in (("g38", "smart_gemini"), ("sonnet", "smart_sonnet")):
        for r in csv.DictReader(open(os.path.join(S, "ideas5", d, "real_results.csv"))):
            c = r["case_id"]
            rows.append((agent, "real", c, proc_of(c), c.split("/")[0], perform[c], 1 - int(r["success"])))
        for r in csv.DictReader(open(os.path.join(S, "ideas5", d, "synth_results.csv"))):
            c = r["synthetic_case_id"]; lab = r.get("perform_label_by_code", r.get("perform_label"))
            rows.append((agent, "synth", c, r["procedure_id"], c.split("/")[0], int(lab), 1 - int(r["success"])))
    for l in open(os.path.join(BASE, "responses.jsonl")):
        j = json.loads(l)
        for c, y in j["responses"].items():
            if y is None: continue
            rows.append(("pub:" + j["subject_id"], "public", c, proc_of(c), c.split("/")[0], perform[c], 1 - int(y)))
    df = pd.DataFrame(rows, columns=["agent", "fid", "case_id", "proc", "domain", "perform", "fail"])
    df = df[~df.proc.isin(BROKEN)].reset_index(drop=True)
    # zero-shot text prior: Sonnet pre-mortem difficulty per real case (ideas10/fc), averaged per procedure and class
    fc = pd.read_csv(os.path.join(S, "ideas10", "fc", "fc_sopbench.csv"))
    fc["proc"] = fc.case_id.map(proc_of); fc["perform"] = fc.case_id.map(perform)
    prior = fc.groupby(["proc", "perform"]).fc_diff.mean().rename("text_prior").reset_index()
    return df, prior


if __name__ == "__main__":
    df, prior = build()
    df.to_csv(os.path.join(os.path.dirname(__file__), "outcomes.csv"), index=False)
    prior.to_csv(os.path.join(os.path.dirname(__file__), "text_prior.csv"), index=False)
    print(df.groupby(["fid", "agent"]).agg(n=("fail", "size"), fail=("fail", "mean"), procs=("proc", "nunique")).query("fid != 'public'"))
    print("public agents", df[df.fid == "public"].agent.nunique(), "rows", (df.fid == "public").sum())
    print("procedures", df.proc.nunique(), "domains", df.domain.nunique())
