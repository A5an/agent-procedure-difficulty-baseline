import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json,csv,re
import numpy as np
D=ZR.REPO + "/data/"
def load(b):
    st={}
    for l in open(D+b+'/statements.jsonl'):
        r=json.loads(l); st[r['case_id']]=r['text']
    tk={r['case_id']:r for r in csv.DictReader(open(D+b+'/tasks.csv'))}
    it={r['']:float(r['b']) for r in csv.DictReader(open(D+b+'/irt/1d_1pl/items.csv'))}
    return st,tk,it
