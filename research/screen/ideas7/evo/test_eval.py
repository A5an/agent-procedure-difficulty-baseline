import sys, time, numpy as np
sys.path.insert(0, "."); import evaluator as E
c1 = '''def features(r):
    d = r["dryrun"]
    return {"p_refuse": d.get("p_refuse") or 0.0, "nlook": d.get("n_lookups") or 0.0}
'''
c2 = '''def features(r):
    return {"len_msg": len(r["message"]), "ncond": len(r["conditions"]), "ntools": len(r["tools"])}
'''
c3 = '''import random
def features(r):
    random.seed(hash(r["id"]) % 1000)
    return {"noise": random.random()}
'''
c4 = '''def features(r):
    return {"x": 1.0}
'''
c5 = '''def features(r):
    return {"a": open("/etc/passwd").read()}
'''
for n, c in [("dry", c1), ("len", c2), ("noise", c3), ("const", c4), ("evil", c5)]:
    t = time.time(); o = E.score(c); print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k != "per_domain_gain"}, round(time.time() - t, 1))
