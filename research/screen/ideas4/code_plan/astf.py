"""Static AST features of a handler function. Nothing is executed."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import ast, re, json
from pathlib import Path
S = Path(ZR.SCREEN)
TT = json.load(open(S / "ideas3/plan_tools/tool_types.json"))
READ_PAT = re.compile(r"^(get_|show_|view_|find_|check_|internal_|validate_|list_|search_|calculate|login|logout|authenticate|is_)")
COLS = ["cp_calls", "cp_writes", "cp_reads", "cp_ifs", "cp_depth", "cp_cmp", "cp_date", "cp_loops", "cp_early_ret", "cp_refuse_ret",
        "cp_guarded", "cp_guarded_writes", "cp_distinct", "cp_boolops", "cp_lines"]
def is_write(dom, name):
    t = TT.get(f"{dom}/{name}")
    if t is not None: return t == "write"
    return not READ_PAT.match(name)
def strip(src):
    src = re.sub(r"^```(?:python)?\s*|```\s*$", "", src.strip(), flags=re.M).strip()
    return src
def feats(src, dom):
    tree = ast.parse(strip(src))
    f = dict.fromkeys(COLS, 0.0); names = set()
    def tool_name(n):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == "tools":
            return n.func.attr
    def refuse(v):
        if v is None: return True
        if isinstance(v, ast.Constant):
            return v.value is None or v.value is False or (isinstance(v.value, str) and re.search(r"refus|transfer|den|reject|fail|not|error|cannot", v.value, re.I) is not None)
        return False
    def walk(n, depth, guard, in_if):
        for c in ast.iter_child_nodes(n):
            d, g, ii = depth, guard, in_if
            if isinstance(c, (ast.If, ast.For, ast.While, ast.Try, ast.IfExp)):
                d = depth + 1; g = guard or isinstance(c, (ast.If, ast.IfExp, ast.While))
                f["cp_depth"] = max(f["cp_depth"], d)
                if isinstance(c, (ast.If, ast.IfExp)): f["cp_ifs"] += 1; ii = True
                if isinstance(c, (ast.For, ast.While)): f["cp_loops"] += 1
            if isinstance(c, ast.Compare): f["cp_cmp"] += len(c.ops)
            if isinstance(c, ast.BoolOp): f["cp_boolops"] += len(c.values) - 1
            if isinstance(c, ast.Attribute) and c.attr in ("fromisoformat", "strptime", "strftime", "timedelta", "days", "date", "now", "today", "total_seconds", "datetime") or \
               isinstance(c, ast.Name) and c.id in ("datetime", "timedelta", "date"):
                f["cp_date"] += 1
            tn = tool_name(c)
            if tn:
                names.add(tn); f["cp_calls"] += 1
                w = is_write(dom, tn); f["cp_writes" if w else "cp_reads"] += 1
                if guard: f["cp_guarded"] += 1; f["cp_guarded_writes"] += w
            if isinstance(c, ast.Return):
                if in_if: f["cp_early_ret"] += 1
                if refuse(c.value) and in_if or (c.value is None and in_if): f["cp_refuse_ret"] += 1
            walk(c, d, g, ii)
    walk(tree, 0, False, False)
    f["cp_distinct"] = len(names); f["cp_lines"] = len([l for l in strip(src).splitlines() if l.strip()])
    return f
