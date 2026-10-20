import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json,glob,collections,csv,pandas as pd
S=ZR.SCREEN + "/"
R=ZR.REPO + "/data/sopbench/"
st={json.loads(l)['case_id']:json.loads(l)['text'] for l in open(R+'statements.jsonl')}
def split(t):
    pol=t.split('Policy for this action:')[1].split('Customer message:')[0].strip()
    msg=t.split('Customer message:')[1].split('Available tools:')[0].strip()
    return pol,msg
real={r['case_id']:r for r in csv.DictReader(open(S+'ideas5/smart_gemini/real_results.csv'))}
rows=[];vpol={}
for f in sorted(glob.glob(S+'ideas2/emulate_run/sop/data/*_tasks.json')):
    dom=f.split('/')[-1].replace('_tasks.json','')
    d=json.load(open(f))
    for proc,cs in d.items():
        g=collections.OrderedDict()
        for i,c in enumerate(cs):
            cid=f'{dom}/{proc}/{i}'
            if cid not in st: continue
            pol,msg=split(st[cid])
            g.setdefault(json.dumps(c['constraints_original'],sort_keys=True),[]).append(cid)
            rows.append(dict(kind='real',case_id=cid,proc=f'{dom}/{proc}',vi=list(g).index(json.dumps(c['constraints_original'],sort_keys=True)),policy=pol,msg=msg,refuse=1-int(real[cid]['perform_label']) if cid in real else None))
        for vi,(k,L) in enumerate(g.items()): vpol[(f'{dom}/{proc}',vi)]=split(st[L[0]])[0]
for f in sorted(glob.glob(S+'ideas2/emulate_run/synth_all/*.json')):
    dom=f.split('/')[-1].replace('_tasks.json','')
    for proc,cs in json.load(open(f)).items():
        for x in cs:
            p=f'{dom}/{proc}'
            rows.append(dict(kind='synth',case_id=x['_cid'],proc=p,vi=x['_variant'],policy=vpol[(p,x['_variant'])],msg=x['user_prompt'],refuse=1-int(x['action_should_succeed'])))
df=pd.DataFrame(rows)
sy=pd.read_csv(S+'ideas5/smart_gemini/synth_results.csv').set_index('synthetic_case_id')
m=df.kind=='synth'
chk=(1-sy.loc[df[m].case_id,'perform_label_by_code']).values
print('label agree w/ code',(chk==df[m].refuse.values).mean())
df.loc[m,'refuse']=chk
df.to_csv('data.csv',index=False)
print(df.groupby('kind').refuse.agg(['count','mean']),df.proc.nunique())
