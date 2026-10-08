# replay stored agent trajectories through the same evaluator call path and compare with stored evaluations
import sop, json, glob, execu
from execu import try_eval
from env.evaluator import evaluator_function_directed_graph
n=ok=0
for f in sorted(glob.glob(sop.SOP+"/output/*/ast_gpt-4.1-mode_fc-*.json"))+sorted(glob.glob(sop.SOP+"/output/*/ast_gpt-4o-mode_fc-*.json")):
    for rec in json.load(open(f)):
        if not rec.get("evaluations"): continue
        il=rec["interactions"][0]; inter=il["interaction"]; fc=[]
        for i in range(len(inter)-1):
            if inter[i].get("tool_calls",[]):
                tcs=[tc for tc in inter[i]["tool_calls"] if tc["function"]["name"].lower() not in ["n/a","na","none","null"]]
                inter[i]["tool_calls"]=tcs
                if tcs: fc.append({"tool_name":inter[i+1]["tool_name"],"arguments":try_eval(tcs[0]["function"]["arguments"]),"content":try_eval(inter[i+1]["content"])})
        ev=evaluator_function_directed_graph(rec["domain"],rec["task"],inter,fc,{"final_database":il["database"]},"full")
        n+=1; ok+= (ev["success"]==rec["evaluations"][0]["success"] and ev["dirgraph_satisfied"]==rec["evaluations"][0]["dirgraph_satisfied"])
print(n,ok)
