import sys; sys.argv=["x"]
import analyze as A, json, glob, collections
mism=0; n=0
for d in ["bank","hotel"]:
    recs=json.load(open(glob.glob(f"{A.SOPORIG}{d}/ast_gemini-2.5-flash-mode_fc-*.json")[0]))
    for r in recs[:60]:
        ev0=r["evaluations"][0]; ir=r["interactions"][0]
        ev=A.R.evaluate(r["task"], d, ir["interaction"], ir["database"])
        n+=1; mism+= (ev["success"]!=ev0["success"])
print("reevaluated",n,"mismatch",mism)
