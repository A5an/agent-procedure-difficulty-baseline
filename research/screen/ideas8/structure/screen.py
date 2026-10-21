import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json,re,csv,sys
sys.path.insert(0,ZR.SCREEN + "/ideas")
from cpulock import cpu_lock
from scipy.stats import spearmanr
import numpy as np
R=ZR.REPO + "/data/"
def feats(text,bench):
    m=text.split('Customer message:')[1].split('Available tools:')[0] if bench=='sop' else text.split('Customer scenario:')[1]
    nums=re.findall(r'(?<![A-Za-z_])\$?\d[\d,]*\.?\d*',m)
    ids=re.findall(r'\b[a-z]+(?:_[a-z0-9]+)+\b',m)
    codes=re.findall(r'\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{5,8}\b',m)
    fmts=set()
    for t in nums: fmts.add('num_dec' if '.' in t else 'num_int')
    if ids: fmts.add('snake')
    if codes: fmts.add('code')
    if re.search(r'\b\d{4}-\d\d-\d\d\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',m): fmts.add('date')
    if re.search(r'\b\d+:\d\d\b|\b[ap]m\b',m,re.I): fmts.add('time')
    return dict(n_num=len(nums),n_id=len(ids),n_code=len(codes),n_fmt=len(fmts),n_typed=len(nums)+len(ids)+len(codes),mlen=len(m.split()))
with cpu_lock("structure"):
  for bench,d in [('sop','sopbench'),('tau2','tau2')]:
    st={json.loads(l)['case_id']:json.loads(l)['text'] for l in open(R+d+'/statements.jsonl')}
    b={r[0]:float(r[1]) for r in list(csv.reader(open(R+d+'/irt/1d_1pl/items.csv')))[1:]}
    tasks=list(csv.DictReader(open(R+d+'/tasks.csv')))
    dom={t['case_id']:t['domain'] for t in tasks}
    F={c:feats(st[c],bench) for c in st if c in b}
    for k in ['n_num','n_id','n_code','n_fmt','n_typed','mlen']:
        out=[]
        for dm in sorted(set(dom.values())):
            cs=[c for c in F if dom[c]==dm]
            x=[F[c][k] for c in cs]; y=[b[c] for c in cs]
            r=spearmanr(x,y)[0] if len(set(x))>1 else float('nan')
            out.append((dm,len(cs),round(r,3)))
        print(bench,k,out,'mean',round(float(np.nanmean([o[2] for o in out])),3))
