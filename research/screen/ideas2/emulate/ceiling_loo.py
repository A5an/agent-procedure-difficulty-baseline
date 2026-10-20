exec(open("ceiling.py").read().split("cheap=[")[0])
out={}
for a in ["gpt-4.1-mini-mode_fc","gemini-2.5-flash-mode_fc","gemma-4-e4b-it-mode_fc","qwen3.5-4b-mode_fc","gpt-4o-mini-mode_fc"]:
    others=[x for x in agents if x!=a]
    tgt=F[others].mean(axis=1).groupby([t.domain,t.procedure_id]).mean()
    out[a]={"vs_mean_failure_of_other_27":pair_rho(est_full(a),tgt)}
print(out)
json.dump(out,open("ceiling_loo.json","w"),indent=1)
