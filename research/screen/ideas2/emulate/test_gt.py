import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, glob, collections
sys.path.insert(0,ZR.SCREEN + "/ideas2/emulate")
import gt
SOP=gt.SOP
agree=collections.Counter(); bad=[]
for dom in ["bank","dmv","healthcare","hotel","library","online_market","university"]:
    t=json.load(open(f"{SOP}/data/{dom}_tasks.json"))
    for p,cs in t.items():
        for i,c in enumerate(cs):
            try:
                lab,_=gt.leaf_results(dom,c,goal=p)
            except Exception as e:
                agree[(dom,'err')]+=1; bad.append((dom,p,i,repr(e)[:100])); continue
            agree[(dom,lab==c['action_should_succeed'])]+=1
            if lab!=c['action_should_succeed']: bad.append((dom,p,i,lab))
print(sorted(agree.items(),key=str)); print(bad[:15])
