import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys,json,hashlib
sys.path.insert(0,ZR.SCREEN + "/ideas2/compile")
sys.path.insert(0,ZR.SCREEN + "/ideas6/msg_facts")
import sop, extract, llmc
assert "msg_facts" in llmc.CACHE
cs=sop.load_cases(); out={}
for t in cs:
    p=extract.PROMPT.format(domain=t["domain"].replace("_"," "),names=extract.names_for(t),msg=t["user_prompt"])
    key=hashlib.sha256(f"0.0|low|msgfacts|{p}".encode()).hexdigest()
    if key in llmc._mem: out[t["case_id"]]=extract.parse(llmc._mem[key]["text"])
print(len(out)); json.dump(out,open(ZR.SCREEN + "/ideas6/msg_facts/facts_partial.json","w"))
