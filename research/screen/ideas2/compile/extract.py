import sys, json, os, re
from concurrent.futures import ThreadPoolExecutor
import sop, common, llmc
HERE=os.path.dirname(os.path.abspath(__file__))
PROMPT="""A customer wrote the message below to a {domain} service desk. List the facts the customer states that a clerk would need as inputs, as one JSON object.

Use these parameter names for the keys (only those the customer actually provides, omit the rest):
{names}

Rules: copy values exactly as written, keep numbers as JSON numbers, and use a nested object for a driver's license identification ({{"drivers_license_id": ..., "drivers_license_state": ...}}) or any other structured value. Do not infer anything that is not stated. Return only the JSON object.

Customer message:
\"\"\"{msg}\"\"\"
"""
_names={}
def names_for(t):
    d=t["domain"]
    if d not in _names:
        _,ai=sop.agent_view(t); seen={}
        for a in ai["tools"]:
            for k,v in a["function"]["parameters"]["properties"].items():
                if k not in seen: seen[k]=v.get("description","")
        _names[d]="\n".join(f"- {k}: {v}" for k,v in seen.items())
    return _names[d]
def parse(text):
    m=re.search(r"\{.*\}",text,re.S)
    try: return json.loads(m.group(0))
    except Exception: return None
def run(domains=None):
    cs=[t for t in sop.load_cases() if not domains or t["domain"] in domains]
    path=os.path.join(HERE,"params_B.json"); res=json.load(open(path)) if os.path.exists(path) else {}
    todo=[t for t in cs if t["case_id"] not in res]
    def f(t):
        p=PROMPT.format(domain=t["domain"].replace("_"," "),names=names_for(t),msg=t["user_prompt"])
        text,u=llmc.gen(p,max_tokens=1500,thinking="low",tag="extract")
        return t["case_id"],parse(text)
    with ThreadPoolExecutor(8) as ex:
        for k,v in ex.map(f,todo): res[k]=v
    json.dump(res,open(path,"w"),indent=1); print(len(todo),llmc.STATS)
if __name__=="__main__":
    run(sys.argv[1].split(",") if len(sys.argv)>1 and sys.argv[1]!="all" else None)
