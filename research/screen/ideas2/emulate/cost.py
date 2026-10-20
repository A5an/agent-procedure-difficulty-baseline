"""Itemised cost of the full emulated-pilot experiment. Token counts per episode come from the logged SOPBench runs (E8_PLAN section 3.2),
prices from E8_PLAN section 2 (Vertex standard, read 4 Oct 2026). Nothing here was run."""
import json
VARIANTS=239; PROCS=70
EP_IN=37653; EP_OUT_VIS=275; CALLS=6.3          # gemini-2.5-flash episode, mean over 830 SOPBench cases
price={ # usd per 1M tokens (input, output incl. thinking)
 "gemini-2.5-flash-lite":(0.10,0.40), "gemini-2.5-flash":(0.30,2.50), "gemini-3.8-flash (until 31 Dec 2026)":(0.75,3.75),
 "gemini-3.8-flash (from 1 Jan 2027)":(1.50,7.50), "gemini-2.5-pro":(1.25,10.0)}
def ep_cost(model, think=500, cache=0.0):
    pin,pout=price[model]; inp=EP_IN*(1-0.9*cache)
    return (inp*pin+(EP_OUT_VIS+think*CALLS)*pout)/1e6
out={"episode_usd":{}}
for m in price:
    out["episode_usd"][m]={"no_thinking":round(ep_cost(m,0),4),"central_500":round(ep_cost(m,500),4),"high_1500":round(ep_cost(m,1500),4),"central_with_70pct_cache_hit":round(ep_cost(m,500,0.7),4)}
gen_call=0.0184   # measured: 110 calls, mean 3.0k input and 4.3k output tokens at 0.75/3.75
def scenario(name, n_variants, cases_per_variant, agents, attempts=1, gen_calls_per_variant=1, margin=0.2, think=500):
    cases=n_variants*cases_per_variant; eps=cases*attempts
    gen=n_variants*gen_calls_per_variant*gen_call
    rows={a:round(eps*ep_cost(a,think),2) for a in agents}
    tot=(gen+sum(rows.values()))*(1+margin)
    calls=sum(eps*CALLS for a in agents)
    # wall clock: 5 s per call for flash class, 8 parallel workers
    hrs=calls*5/8/3600
    return {"variants":n_variants,"cases_per_variant":cases_per_variant,"synthetic_cases":cases,"episodes_per_agent":eps,"agents":agents,
            "generation_usd":round(gen,2),"episode_usd":rows,"total_with_20pct_margin":round(tot,2),"model_calls":int(calls),"wall_hours_8_workers":round(hrs,1)}
out["scenarios"]={
 "S0 smoke test: 10 procedures x modal variant, 10 cases, flash-lite":scenario("S0",10,10,["gemini-2.5-flash-lite"]),
 "S1 minimal: 70 procedures x modal variant, 20 cases, one cheap agent":scenario("S1",70,20,["gemini-2.5-flash-lite"],gen_calls_per_variant=3),
 "S2 main: all 239 policy variants x 6 cases, two agents (flash-lite and 2.5-flash)":scenario("S2",239,6,["gemini-2.5-flash-lite","gemini-2.5-flash"]),
 "S3 strong: S2 plus gemini-3.8-flash, 2 attempts":scenario("S3",239,6,["gemini-2.5-flash-lite","gemini-2.5-flash","gemini-3.8-flash (until 31 Dec 2026)"],attempts=2),
}
json.dump(out,open("cost.json","w"),indent=1); print(json.dumps(out,indent=1))
