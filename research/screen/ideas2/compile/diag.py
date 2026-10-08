import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, sop, common, execu
progs=json.load(open(ZR.SCREEN + '/ideas2/compile/programs_t0.json'))
cs={t['case_id']:t for t in sop.load_cases()}
rows=json.load(open(ZR.SCREEN + '/ideas2/compile/results_A.json'))
def show(case_id,code=False):
    t=cs[case_id]; k=common.key_of(t); ev,log,err=execu.execute(progs[k]['code'],t['user_known'],t)
    print("==",case_id,"should",t['action_should_succeed'],"success",ev['success'],err)
    print(" known",json.dumps(t['user_known'])[:300])
    for c in log: print("  ",c['tool_name'],json.dumps(c['arguments'])[:140],"->",str(c['content'])[:140])
    if code: print(progs[k]['code'])
if __name__=="__main__":
    proc=sys.argv[1]; n=int(sys.argv[2]); c=0
    for r in rows:
        if r['procedure']==proc and r['should'] and not r['success']:
            show(r['case_id'],code=(c==0 and len(sys.argv)>3)); c+=1
            if c>=n: break
