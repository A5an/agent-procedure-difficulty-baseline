import sop, copy, json, signal, ast, traceback
from env.variables import domain_keys
from env.evaluator import evaluator_function_directed_graph
def try_eval(x):
    try: return eval(x)
    except Exception: return x
class Timeout(Exception): pass
def _alarm(*a): raise Timeout()
class Tools:
    def __init__(self,ds,allowed,maxcalls=30):
        self.ds=ds; self.allowed=allowed; self.log=[]; self.maxcalls=maxcalls
    def __getattr__(self,name):
        if name.startswith("__"): raise AttributeError(name)
        def call(*args,**kw):
            if args: return "Error: positional arguments not supported"
            if len(self.log)>=self.maxcalls: raise Timeout()
            if name not in self.allowed or not hasattr(self.ds,name):
                content=f"Error: Tool {name} not found."
                self.log.append({"tool_name":name,"arguments":kw,"content":content}); return content
            try:
                raw=getattr(self.ds,name)(**kw)
                if isinstance(raw,tuple): raw=raw[1]
            except Exception as e:
                raw=f"{e.__class__.__name__}: {str(e)}"
            s=str(raw)  # same stringification as swarm
            self.log.append({"tool_name":name,"arguments":kw,"content":try_eval(s)})
            return try_eval(s)
        return call
def execute(code,params,t,default_cache={}):
    d=t["domain"]
    inn,full,descr=sop.defaults(d)
    ds=domain_keys[d](copy.deepcopy(t["initial_database"]),inn,t["constraint_parameters"])
    _,ai=sop.agent_view(t)
    allowed={a["function"]["name"] for a in ai["tools"]}
    tools=Tools(ds,allowed)
    err=None
    try:
        g={"__builtins__":__builtins__}
        exec(code,g)
        signal.signal(signal.SIGALRM,_alarm); signal.alarm(5)
        try: g["handle"](copy.deepcopy(params),tools)
        finally: signal.alarm(0)
    except Timeout: err="timeout_or_toomany_calls"
    except BaseException as e: err=f"{e.__class__.__name__}: {str(e)[:200]}"
    ev=evaluator_function_directed_graph(d,t,[],tools.log,{"final_database":ds.data},"full")
    return ev,tools.log,err
