"""Analysis only (uses gold constraints): what makes must-perform cases hard on SOPBench, within procedure."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import pandas as pd, numpy as np, re, json
from scipy.stats import spearmanr
from ceil import load
s=load("sopbench")
R=ZR.DATA + "/SOPBench-d2622008/data"
raw={d:json.load(open(f"{R}/{d}_tasks.json")) for d in s.domain.unique()}
def cons(cid):
    d,a,i=cid.split("/"); return raw[d][a][int(i)]
rows=[]
for _,r in s.iterrows():
    e=cons(r.case_id); c=json.dumps(e["constraints"]); g=e["directed_action_graph"]["nodes"]
    pol=re.search(r"Policy for this action:\n(.*?)\nCustomer message:", r.text, re.S).group(1)
    rows.append(dict(case_id=r.case_id, n_single=c.count('"single"'), n_or=c.count('"or"'), n_chain=c.count('"chain"'),
        n_gold_calls=sum(1 for n in g if isinstance(n,list)), pol_bullets=pol.count("•"), pol_any=pol.count("ANY ONE"),
        pol_time=len(re.findall(r"interaction|hours|days after|days before",pol)), pol_len=len(pol),
        msg_len=len(e["user_prompt"]), n_known=len(e["user_known"])))
f=pd.DataFrame(rows).set_index("case_id"); s=s.join(f,on="case_id")
def within(df,col):
    out=[]
    for p,g in df.groupby("procedure_id"):
        if len(g)>=4 and g[col].nunique()>1: out.append((spearmanr(g[col],g.b).correlation,len(g)))
    if not out: return np.nan, 0
    r=np.array(out); return float(np.average(r[:,0],weights=r[:,1])), len(out)
for lab in [1,0]:
    d=s[s.should_succeed==lab]
    print("label",lab,"cases",len(d))
    for col in f.columns: print("   %-14s within-proc rho %+.3f (procs %d)"%((col,)+within(d,col)))

print("---- message-visible proxies (exec / refuse) ----")
tc=pd.read_csv("../ideas8/textcount/txt_sopbench.csv").set_index("case_id")
s=s.join(tc,on="case_id")
s["msg"]=s.case_id.map(lambda c: cons(c)["user_prompt"])
s["n_quoted"]=s.msg.str.count(r'"[^"]+"')
s["has_pw"]=s.msg.str.contains("password|identification|license",case=False).astype(int)
s["has_admin"]=s.msg.str.contains("admin",case=False).astype(int)
keys=lambda c: list(cons(c)["user_known"].keys())
s["known_keys"]=s.case_id.map(lambda c: ",".join(sorted(keys(c))))
for col in ["dig_chars","dig_nums","dig_ids","cred_word","n_quoted","has_pw","has_admin"]:
    print("   %-10s exec %+.3f  refuse %+.3f   corr with n_known (exec) %+.2f"%(col,within(s[s.should_succeed==1],col)[0],within(s[s.should_succeed==0],col)[0], spearmanr(s[s.should_succeed==1][col],s[s.should_succeed==1].n_known).correlation))
e=s[s.should_succeed==1]
for p,g in e.groupby("procedure_id"):
    if g.n_known.nunique()>1 and len(g)>=4:
        print(p, [(k, round(b,1)) for k,b in zip(g.known_keys,g.b)][:8])

print("---- all cases ----")
s["pol"]=s.text.map(lambda x: re.search(r"Policy for this action:\n(.*?)\nCustomer message:", x, re.S).group(1))
s["pol_login"]=s.pol.str.contains("logged in|credentials|authenticat|password",case=False).astype(int)
s["nocred"]=1-s.cred_word
s["nocred_nologin"]=((s.cred_word==0)&(s.pol_login==0)).astype(int)
s["nocred_login"]=((s.cred_word==0)&(s.pol_login==1)).astype(int)
for col in ["nocred","has_admin","nocred_nologin","nocred_login","pol_login"]:
    print("   %-15s all %+.3f exec %+.3f refuse %+.3f   n=%d"%(col,within(s,col)[0],within(s[s.should_succeed==1],col)[0],within(s[s.should_succeed==0],col)[0], s[col].sum()))
print(s.groupby(["should_succeed","pol_login","cred_word"]).b.agg(["mean","count"]).round(2))

print("---- pol_login by domain (within-proc rho, all cases) ----")
for d,g in s.groupby("domain"):
    print("  ",d, "%+.3f"%within(g,"pol_login")[0], "n_procs_var", within(g,"pol_login")[1], "share with login %.2f"%g.pol_login.mean())
