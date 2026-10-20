import sys, numpy as np, json
sys.path.insert(0, "."); import evaluator as E
res = []
for k in range(40):
    c = f'''def features(r):
    s = 0
    for i, ch in enumerate(r["id"]): s = (s * {31 + 2*k} + ord(ch) * {7 + k}) % 100003
    s2 = (s * 7919 + {k*13}) % 10007
    return {{"n1": s / 100003.0, "n2": s2 / 10007.0}}
'''
    o = E.score(c); res.append(o["score"])
res = np.array(res); print("null gains: mean %.4f sd %.4f min %.4f max %.4f p95 %.4f" % (res.mean(), res.std(), res.min(), res.max(), np.percentile(res, 95)))
json.dump(res.tolist(), open("null_calibration.json", "w"))
