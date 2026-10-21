"""Step 1: one gemini-3.8-flash call per case. Reads ONLY the customer message and the domain's parameter-name vocabulary.
Output: typed stated facts. No perform/refuse judgement."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, os, re
HERE_=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,ZR.SCREEN + "/ideas2/compile")
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,HERE_)
import sop, llmc
HERE=os.path.dirname(os.path.abspath(__file__))
PROMPT="""A customer wrote the message below to a {domain} service desk. Extract every concrete fact the customer states about themselves, their account, their records or their request: amounts, numbers, dates, times, identifiers, names, requested options, and statements about their own status (for example an age, a balance, a membership, a past event).

Output one JSON object with two keys.
"params": an object whose keys are taken from this vocabulary of parameter names (use a key only if the customer actually provides that value, omit the rest):
{names}
"other": a list of objects {{"field": <short snake_case name of the thing>, "value": <value>, "type": <"number"|"date"|"string"|"boolean">}} for stated facts that do not fit a parameter name above.

Rules: copy values exactly as written, keep numbers as JSON numbers, dates as written, and use a nested object for a driver's license identification ({{"drivers_license_id": ..., "drivers_license_state": ...}}). Do not infer anything that is not stated. Do not judge whether the request is allowed or will succeed. Return only the JSON object.

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
def run(limit=None):
    cs=sop.load_cases()[:limit]
    path=os.path.join(HERE,"facts.json"); res=json.load(open(path)) if os.path.exists(path) else {}
    todo=[t for t in cs if res.get(t["case_id"]) is None]
    def f(t):
        p=PROMPT.format(domain=t["domain"].replace("_"," "),names=names_for(t),msg=t["user_prompt"])
        text,u=llmc.gen(p,max_tokens=2000,thinking="low",tag="msgfacts")
        return t["case_id"],parse(text)
    with ThreadPoolExecutor(6) as ex:
        for k,v in ex.map(f,todo): res[k]=v
    json.dump(res,open(path,"w"),indent=1); print(len(todo),llmc.STATS,sum(v is None for v in res.values()),"unparsed")
if __name__=="__main__":
    run(int(sys.argv[1]) if len(sys.argv)>1 else None)
