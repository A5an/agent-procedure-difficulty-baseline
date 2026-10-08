"""Two extra temperature-1 compilations (samples 3 and 4) with the exact compile.py prompt (base variant), one call per policy unit."""
import sys, json, os, re
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
import sop, common, llmc
HERE=os.path.dirname(os.path.abspath(__file__))
def extract_code(text):
    m=re.findall(r"```(?:python)?\n(.*?)```",text,re.S)
    return (m[0] if m else text).strip()
def units():
    u={}
    for t in sop.load_cases(): u.setdefault(common.key_of(t),t)
    return u
def run(sample):
    u=units(); path=os.path.join(HERE,f"programs_t1_s{sample}.json")
    res=json.load(open(path)) if os.path.exists(path) else {}
    todo=[k for k in u if k not in res]
    def f(k):
        try:
            text,usage=llmc.gen(common.compile_prompt(u[k]),temperature=1.0,max_tokens=8000,tag=f"compile{sample}")
            return k,{"code":extract_code(text),"finish":usage.get("finishReason")}
        except Exception as e: return k,None
    with ThreadPoolExecutor(8) as ex:
        for k,r in ex.map(f,todo):
            if r: res[k]=r
    json.dump(res,open(path,"w"),indent=1); print(sample,len(u),"units",len(todo),"todo",llmc.STATS,flush=True)
if __name__=="__main__":
    for s in (3,4): run(s)
