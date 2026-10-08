import sys, json, os, collections, re
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
import sop, common, llmc
HERE=os.path.dirname(os.path.abspath(__file__))
def extract_code(text):
    m=re.findall(r"```(?:python)?\n(.*?)```",text,re.S)
    return (m[0] if m else text).strip()
def units(domains=None):
    cs=sop.load_cases(); u={}
    for t in cs:
        if domains and t["domain"] not in domains: continue
        u.setdefault(common.key_of(t),t)
    return u
def run(domains=None,temperature=0.0,sample=0,out="programs_t0.json",variant="base"):
    u=units(domains); path=os.path.join(HERE,out)
    res=json.load(open(path)) if os.path.exists(path) else {}
    todo=[k for k in u if k not in res]
    def f(k):
        text,usage=llmc.gen((common.compile_prompt_C(u[k]) if variant=='C' else common.compile_prompt(u[k])),temperature=temperature,max_tokens=8000,tag=f"compile{sample}"+("C" if variant=="C" else ""))
        return k,{"code":extract_code(text),"in":usage.get("promptTokenCount",0),"out":usage.get("candidatesTokenCount",0)+usage.get("thoughtsTokenCount",0),"finish":usage.get("finishReason")}
    with ThreadPoolExecutor(8) as ex:
        for k,r in ex.map(f,todo): res[k]=r
    json.dump(res,open(path,"w"),indent=1); print(len(u),"units",len(todo),"new",llmc.STATS)
if __name__=="__main__":
    run(domains=sys.argv[1].split(",") if len(sys.argv)>1 and sys.argv[1]!="all" else None)
