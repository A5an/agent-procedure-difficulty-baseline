import json, gen, gt, collections
res={}
for d,p in gen.SEL:
    cs,same=gen.modal_variant(d,p)
    gs=collections.Counter(json.dumps(c["directed_action_graph"],sort_keys=True) for c in same)
    byl=collections.defaultdict(set)
    for c in same: byl[c["action_should_succeed"]].add(json.dumps(c["directed_action_graph"],sort_keys=True))
    res[f"{d}.{p}"]=(len(same),len(gs),{k:len(v) for k,v in byl.items()})
    # constraint variants -> graph
    allg=collections.defaultdict(set)
    for c in cs: allg[json.dumps(c["constraints"],sort_keys=True)].add(json.dumps(c["directed_action_graph"],sort_keys=True))
    res[f"{d}.{p}"]+= (max(len(v) for v in allg.values()),)
print(res)
