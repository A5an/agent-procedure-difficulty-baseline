import json, copy, inspect, gt, gen, validate
res={}
for dom,proc in gen.SEL:
    cs=json.load(open(f"{gt.SOP}/data/{dom}_tasks.json"))[proc]
    ok=0; ld=0
    for c in cs:
        l,_=gt.load_check(dom,c,goal=proc); ld+=l
        try:
            dep_innate, default_dep, _ = gt.defaults(dom)
            dep_full = copy.deepcopy(default_dep); dep_full[proc] = c["constraints"]
            st = gt.domain_keys[dom + "_strict"](copy.deepcopy(c["initial_database"]), dep_innate, dep_full, c["constraint_parameters"])
            uk=c["user_known"]
            params=[p for p,v in inspect.signature(getattr(st.evaluation_get_domain_system(),proc)).parameters.items() if v.default is inspect.Parameter.empty and v.kind not in (v.VAR_POSITIONAL,v.VAR_KEYWORD)]
            if "username" in uk:
                lp=[p for p in inspect.signature(st.login_user).parameters]
                if all(p in uk for p in lp): st.login_user(**{p:uk[p] for p in lp})
            getattr(st,proc)(**{p:uk[p] for p in params}); ok+=1
        except Exception as e: print(dom,proc,repr(e)[:80])
    res[f"{dom}.{proc}"]={"n":len(cs),"loads":ld,"tool_runs":ok}
print(res); json.dump(res,open("real_validity.json","w"))
