import json
tp=tc=th=n=0
for l in open("cache_gen.jsonl"):
    r=json.loads(l); u=r["usage"] or {}
    tp+=u.get("promptTokenCount",0); tc+=u.get("candidatesTokenCount",0); th+=u.get("thoughtsTokenCount",0); n+=1
cost=lambda pin,pout:(tp*pin+(tc+th)*pout)/1e6
out={"calls_generation":n,"prompt_tokens":tp,"output_tokens":tc,"thought_tokens":th,
     "usd_at_0.75_3.75_until_31_dec_2026":round(cost(0.75,3.75),3),"usd_at_1.50_7.50_after":round(cost(1.5,7.5),3),
     "usd_per_call_mean":round(cost(0.75,3.75)/n,4),"prompt_tokens_mean":tp/n,"output_tokens_mean":(tc+th)/n}
print(out); json.dump(out,open("gen_cost.json","w"),indent=1)
