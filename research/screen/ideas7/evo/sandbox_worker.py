"""Restricted runner for candidate feature functions. usage: sandbox_worker.py records.jsonl < code ; prints JSON {"rows": [...], "keys": [...], "errors": n, "err_example": str}.
Static checks: <= 80 lines, imports only from a small whitelist, no dunder attributes, no open/exec/eval/compile/getattr/setattr/globals/locals.
Run under a per-record timer."""
import sys, json, ast, signal, copy, math, builtins
ALLOWED_MODS = {"math", "re", "json", "statistics", "collections", "itertools", "string"}
BAN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "getattr", "setattr", "delattr", "vars", "dir", "breakpoint", "exit", "quit", "help", "memoryview", "type", "super", "object"}
SAFE = ["abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list", "map", "max", "min", "range",
        "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "frozenset", "pow", "divmod", "ord", "chr", "repr", "hash", "iter", "next", "slice",
        "Exception", "ValueError", "KeyError", "ZeroDivisionError", "TypeError", "IndexError", "AttributeError", "True", "False", "None", "OverflowError"]
def static_check(code):
    if len(code.splitlines()) > 80: return "too many lines"
    tree = ast.parse(code)
    has = False
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "features": has = True
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] not in ALLOWED_MODS: return f"import {a.name} not allowed"
        if isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] not in ALLOWED_MODS: return f"import from {n.module} not allowed"
        if isinstance(n, ast.Attribute) and n.attr.startswith("__"): return "dunder attribute"
        if isinstance(n, ast.Name) and (n.id in BAN_NAMES or n.id.startswith("__")): return f"name {n.id} not allowed"
        if isinstance(n, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)): return "global/async not allowed"
    return None if has else "no features function"
def make_ns():
    import importlib
    def imp(name, g=None, l=None, fromlist=(), level=0):
        if name.split(".")[0] not in ALLOWED_MODS: raise ImportError(name)
        return importlib.import_module(name)
    b = {k: getattr(builtins, k) for k in SAFE if hasattr(builtins, k)}
    b["__import__"] = imp
    return {"__builtins__": b}
class TO(Exception): pass
def _h(sig, frm): raise TO()
def main():
    path = sys.argv[1]; code = sys.stdin.read()
    err = static_check(code)
    if err: print(json.dumps({"static_error": err})); return
    ns = make_ns()
    try: exec(compile(code, "<cand>", "exec"), ns)
    except Exception as e: print(json.dumps({"static_error": "exec: " + repr(e)[:200]})); return
    fn = ns["features"]; signal.signal(signal.SIGALRM, _h)
    rows, keys, errors, ex = [], [], 0, ""
    for l in open(path):
        r = json.loads(l); out = None
        try:
            signal.setitimer(signal.ITIMER_REAL, 1.0)
            d = fn(copy.deepcopy(r))
            signal.setitimer(signal.ITIMER_REAL, 0)
            if not isinstance(d, dict): raise TypeError("features must return a dict")
            out = {}
            for k, v in list(d.items())[:8]:
                v = float(v); out[str(k)] = v if math.isfinite(v) else None
        except BaseException as e:
            signal.setitimer(signal.ITIMER_REAL, 0)
            errors += 1; ex = ex or (type(e).__name__ + ": " + str(e)[:150])
        rows.append({"id": r["id"], "f": out})
    print(json.dumps({"rows": rows, "errors": errors, "err_example": ex}))
main()
