import sys; sys.argv=["x"]
import analyze as A, json, collections, numpy as np
for mo in ["gemini-2.5-flash","gemini-3.8-flash"]:
    h0=A.load_arm(mo,"H0"); 
    recs={}
    for l in open(f"{A.R.OUT}/{mo}__H2.jsonl"):
        r=json.loads(l)
        if not r["error"]: recs[r["case_id"]]=r
    recs=list(recs.values())
    h2=A.load_arm(mo,"H2",strip=True)
    nb=0; bl=[]; rec_ok=0; misl=collections.Counter()
    for r in recs:
        if r.get("blocks"):
            bl.append(r["case_id"]); 
    should=lambda c: h2[c]["should"]
    print(mo,"episodes with >=1 block",len(bl),"of",len(recs),"| should-perform:",sum(should(c) for c in bl),"should-refuse:",sum(not should(c) for c in bl))
    print("  blocked cases: H0 success",np.mean([h0[c]["success"] for c in bl]),"H2 success",np.mean([h2[c]["success"] for c in bl]))
    print("  blocked cases that then performed action:",sum(h2[c]["performed"] for c in bl))
    # H2 failures on should-perform: block that never recovered
    print("  H2 wrongful refusals:",[c for c in h2 if h2[c]["wrongful"]][:40])
    # unnecessary required tools: perform cases blocked
    tot=sum(len(r["blocks"]) for r in recs); print("  total blocks",tot)
# H0 failure types
for mo in ["gemini-2.5-flash","gemini-3.8-flash"]:
    for ar in ["H0","H1","H2"]:
        d=A.load_arm(mo,ar,strip=(ar=="H2"))
        f=[c for c in d if not d[c]["success"]]
        print(mo,ar,"fails",len(f),"unsafe",sum(d[c]["unsafe"] for c in f),"wrongful",sum(d[c]["wrongful"] for c in f),"dirgraph_fail",sum((not d[c]["dirgraph"]) for c in f),
              "other",sum(1 for c in f if not d[c]["unsafe"] and not d[c]["wrongful"]))
