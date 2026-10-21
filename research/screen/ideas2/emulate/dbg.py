import json, gt
SOP=gt.SOP
for p in ["exchange_product"]:
  for i,c in enumerate(json.load(open(f"{SOP}/data/online_market_tasks.json"))[p]):
    lab,res=gt.leaf_results("online_market",c,goal=p)
    if lab!=c['action_should_succeed']:
        print(i,c['user_known'],json.dumps(c['constraints']),res, c['user_prompt'][:200]); break
