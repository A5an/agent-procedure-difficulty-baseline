import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import csv, json, random, collections
B=ZR.SCREEN + "/../"
orig=list(csv.DictReader(open(B+"screen/router_pilot/sample150.csv")))
bad=set(c for c,_ in json.load(open(B+"screen/ideas2/compile/unwinnable_perform_cases.json")))
EXC_PROC={("bank","cancel_credit_card"),("bank","pay_bill_with_credit_card")}
D=ZR.DATA + "/SOPBench-d2622008/data/"
pool=[]
for d in ["bank","dmv","healthcare","hotel","library","online_market","university"]:
    for goal,arr in json.load(open(D+f"{d}_tasks.json")).items():
        for i,t in enumerate(arr):
            pool.append(dict(case_id=f"{d}/{goal}/{i}",domain=d,procedure=goal,carry=str(bool(t["action_should_succeed"]))))
pm={r["case_id"]:r for r in pool}
bad_ok=lambda r: (r["domain"],r["procedure"]) not in EXC_PROC and r["case_id"] not in bad
keep=[r for r in orig if bad_ok(r)]
print("orig",len(orig),"kept",len(keep))
need=collections.Counter((r["domain"],r["carry"]) for r in orig)-collections.Counter((r["domain"],r["carry"]) for r in keep)
have={r["case_id"] for r in keep}
rng=random.Random(20261005)
for (d,c),n in sorted(need.items()):
    cand=[r for r in pool if r["domain"]==d and r["carry"]==c and bad_ok(r) and r["case_id"] not in have and r["case_id"] not in {o["case_id"] for o in orig}]
    rng.shuffle(cand); add=cand[:n]
    print(d,c,"need",n,"pool",len(cand))
    keep+=add
    have|={r["case_id"] for r in add}
keep.sort(key=lambda r:r["case_id"])
with open("sample.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["case_id","domain","procedure","carry"]); w.writeheader(); w.writerows(keep)
print(len(keep),collections.Counter(r["carry"] for r in keep))
