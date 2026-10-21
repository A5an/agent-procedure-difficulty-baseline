"""Cost of the tau2 variant. Tokens per episode are measured in the public gemini-3-flash trajectories (agent and user simulator),
prices are Vertex standard (E8_PLAN section 2). Judge and scenario generation priced at gemini-3.8-flash 0.75/3.75."""
import json
ep={"airline":dict(a_in=59220,a_out=6229,u_in=4395,u_out=432),"retail":dict(a_in=48741,a_out=3549,u_in=3769,u_out=335),"telecom":dict(a_in=138047,a_out=8032,u_in=71351,u_out=855)}
P={"flash-lite":(0.10,0.40),"flash":(0.30,2.50),"3.8-flash":(0.75,3.75)}
c=lambda m,i,o:(i*P[m][0]+o*P[m][1])/1e6
res={}
for d,e in ep.items():
    res[d]={"agent_flash_lite":round(c("flash-lite",e["a_in"],e["a_out"]),4),"agent_flash":round(c("flash",e["a_in"],e["a_out"]),4),
            "user_sim_flash_lite":round(c("flash-lite",e["u_in"],e["u_out"]),4),"user_sim_3.8_flash":round(c("3.8-flash",e["u_in"],e["u_out"]),4),
            "judge_3.8_flash (12k in, 400 out)":round(c("3.8-flash",12000,400),4),"scenario_generation_per_scenario (3k in, 1.5k out)":round(c("3.8-flash",3000,1500),4)}
N=100
tot=0
for d in ep: tot+=N*(res[d]["agent_flash_lite"]+res[d]["user_sim_flash_lite"]+res[d]["judge_3.8_flash (12k in, 400 out)"]+res[d]["scenario_generation_per_scenario (3k in, 1.5k out)"])
tot2=0
for d in ep: tot2+=N*(res[d]["agent_flash"]+res[d]["user_sim_flash_lite"]+res[d]["judge_3.8_flash (12k in, 400 out)"]+res[d]["scenario_generation_per_scenario (3k in, 1.5k out)"])
res["total_300_scenarios_flash_lite_agent"]=round(tot,2); res["total_300_scenarios_flash_agent"]=round(tot2,2)
json.dump(res,open("cost_tau2.json","w"),indent=1); print(json.dumps(res,indent=1))
