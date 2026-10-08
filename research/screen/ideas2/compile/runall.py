import sys, json, os, collections
import sop, common, execu
HERE=os.path.dirname(os.path.abspath(__file__))
def run(variant,progfile="programs_t0.json",out=None,domains=None):
    progs=json.load(open(os.path.join(HERE,progfile)))
    pB=json.load(open(os.path.join(HERE,"params_B.json"))) if variant=="B" else None
    rows=[]
    for t in sop.load_cases():
        if domains and t["domain"] not in domains: continue
        k=common.key_of(t)
        if k not in progs: continue
        params=t["user_known"] if variant=="A" else (pB.get(t["case_id"]) or {})
        ev,log,err=execu.execute(progs[k]["code"],params,t)
        called=any(c["tool_name"]==t["user_goal"] for c in log)
        rows.append({"case_id":t["case_id"],"domain":t["domain"],"procedure":t["user_goal"],"unit":k,
            "should":bool(t["action_should_succeed"]),"success":bool(ev["success"]),"action_successfully_called":bool(ev["action_successfully_called"]),
            "action_called_correctly":bool(ev["action_called_correctly"]),"dirgraph_satisfied":bool(ev["dirgraph_satisfied"]),
            "constraint_not_violated":bool(ev["constraint_not_violated"]),"database_match":bool(ev["database_match"]),
            "no_tool_call_error":bool(ev["no_tool_call_error"]),"err":err,"ncalls":len(log),"tools":[c["tool_name"] for c in log]})
    json.dump(rows,open(os.path.join(HERE,out or f"results_{variant}.json"),"w"))
    n=len(rows); print(variant,n,"success",sum(r["success"] for r in rows)/n,"crash",sum(1 for r in rows if r["err"])/n,
        "acc_decision",sum(r["action_called_correctly"] for r in rows)/n)
    return rows
if __name__=="__main__":
    run(sys.argv[1],domains=sys.argv[2].split(",") if len(sys.argv)>2 and sys.argv[2]!="all" else None)
