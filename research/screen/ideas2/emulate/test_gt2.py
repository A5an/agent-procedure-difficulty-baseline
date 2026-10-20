import sys, json, collections
import gt
SOP=gt.SOP
cnt=collections.Counter(); ex={}
for dom in ["bank","dmv","healthcare","hotel","library","online_market","university"]:
    t=json.load(open(f"{SOP}/data/{dom}_tasks.json"))
    for p,cs in t.items():
        for i,c in enumerate(cs):
            try: lab,_=gt.leaf_results(dom,c,goal=p)
            except Exception as e: lab='err:'+repr(e)[:40]
            if lab!=c['action_should_succeed']:
                cnt[(dom,p,str(lab),c['action_should_succeed'])]+=1; ex.setdefault((dom,p),(i))
for k,v in sorted(cnt.items()): print(k,v,ex[k[:2]])
