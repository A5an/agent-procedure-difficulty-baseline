"""Build clean/data_A and clean/data_B (data roots with sopbench cleaned; tau2, tauk_banking symlinked unchanged)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, shutil, os
from pathlib import Path
import pandas as pd
REPO=Path(ZR.REPO + "/data")
HERE=Path(__file__).parent
DROP=["gpt-5-mode_fc","gpt-5-mini-mode_fc","gemini-2.5-pro-mode_fc"]
em=pd.read_csv(HERE/"empty_cases.csv")
bad=set(zip(em[em.mostly_empty==1].agent, em[em.mostly_empty==1].case_id))
rows=[json.loads(l) for l in open(REPO/"sopbench/responses.jsonl")]
stats={}
for var in "AB":
    root=HERE/f"data_{var}"; sop=root/"sopbench"
    if sop.exists(): shutil.rmtree(sop)
    sop.mkdir(parents=True)
    for f in (REPO/"sopbench").iterdir():
        if f.name in ("responses.jsonl","irt"): continue
        if f.is_dir(): shutil.copytree(f, sop/f.name)
        else: shutil.copy(f, sop/f.name)
    out=[]; removed=0
    for r in rows:
        if var=="A":
            if r["subject_id"] in DROP: removed+=len(r["responses"]); continue
            out.append(r)
        else:
            resp={c:v for c,v in r["responses"].items() if (r["subject_id"],c) not in bad}
            removed+=len(r["responses"])-len(resp)
            out.append({**r,"responses":resp})
    with open(sop/"responses.jsonl","w") as f:
        for r in out: f.write(json.dumps(r)+"\n")
    for b in ("tau2","tauk_banking"):
        if not (root/b).exists(): os.symlink(REPO/b, root/b)
    stats[var]=(len(out),removed, sum(len(r["responses"]) for r in out))
    print(var,"agents",len(out),"cells removed",removed,"cells left",stats[var][2])
tot=sum(len(r["responses"]) for r in rows); print("total cells",tot)
bb=em[em.mostly_empty==1].groupby("agent").size(); print(bb)
