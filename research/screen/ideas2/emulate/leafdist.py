"""Do synthetic cases fail the same policy conditions as real cases? Per (procedure, leaf condition) failure rate, synthetic vs real."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, sys, collections, numpy as np
import gt, gen
from scipy.stats import spearmanr
name=sys.argv[1]; variants=set(sys.argv[2].split(","))
rows=[r for r in json.load(open(f"{ZR.SCREEN}/ideas2/emulate/validation_rows_{name}.json")) if r['variant'] in variants and r.get('tool_runs') and r.get('code_label') is not None]
out={}; xs=[]; ys=[]
for d,p in gen.SEL:
    cs,same=gen.modal_variant(d,p)
    leaves=None; realc=collections.Counter(); n=0
    for c in same:
        lab,leaf=gt.leaf_results(d,c,goal=p); leaves=leaves or list(leaf.keys()); n+=1
        for k,v in leaf.items(): realc[k]+= (not v)
    syn=[r for r in rows if r['proc']==p]
    sc=collections.Counter()
    for r in syn:
        for k in r['fail_pat']: sc[k]+=1
    for k in leaves:
        rr=realc[k]/n; ss=sc[k]/max(1,len(syn)); xs.append(rr); ys.append(ss)
        out.setdefault(f"{d}.{p}",{})[k.split('{')[0]]=(round(rr,2),round(ss,2))
xs=np.array(xs); ys=np.array(ys)
res={"n_pairs":len(xs),"spearman":float(spearmanr(xs,ys).correlation),"mean_abs_diff":float(np.mean(np.abs(xs-ys))),
     "never_failed_in_synthetic_but_fails_in_real":int(((ys==0)&(xs>0)).sum()),"failed_in_real":int((xs>0).sum()),
     "mean_fail_rate_real":float(xs.mean()),"mean_fail_rate_synth":float(ys.mean()),"detail":out}
json.dump(res,open(f"{ZR.SCREEN}/ideas2/emulate/leafdist_{name}_{''.join(sorted(variants))}.json","w"),indent=1)
print({k:v for k,v in res.items() if k!="detail"})
