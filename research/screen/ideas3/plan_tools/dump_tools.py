import json
from tau2.domains.airline.environment import get_environment as a
from tau2.domains.retail.environment import get_environment as r
from tau2.domains.telecom.environment import get_environment as t
out={}
for n,f in [("airline",a),("retail",r),("telecom",t)]:
    e=f(); tools=e.get_tools()
    out[n]={"agent":[x.openai_schema for x in tools]}
    try: out[n]["user"]=[x.openai_schema for x in e.get_user_tools()]
    except Exception as ex: out[n]["user"]=[]; print(n,"user",ex)
    print(n,len(out[n]["agent"]),len(out[n]["user"]))
json.dump(out,open("../tools_raw.json","w"),indent=1)
# tool types (for validation only, not shown to the planner)
types={}
for n,f in [("airline",a),("retail",r),("telecom",t)]:
    e=f()
    for x in e.get_tools():
        types[f"{n}/{x.name}"]=str(e.tools.tool_type(x.name).value)
json.dump(types,open("../tool_types.json","w"),indent=1)
print(types)
