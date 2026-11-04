"""Synthetic procedures (PREREG.md in this folder): spec checks, the correct outcome computed by code, tau2 env per case.

A procedure spec (format in SPEC_FORMAT.md) is written by an LLM. Everything that decides whether an agent passed is
computed here from the spec, never taken from the LLM: which records the agent can reach from what the customer knows,
which rule fires, which write calls are correct, and the database end state those calls produce.
"""
import ast, copy, datetime as dt, inspect, json, re
from typing import Any

TODAY = "2026-05-20"
LEVELS = ["upfront", "on_request", "lookup", "insufficient", "helpful_extra", "distracting", "conflicting"]
BEHAVIORS = ["cooperative", "terse", "pushy", "changes_mind"]
BEHAVIORS_V2 = BEHAVIORS + ["negotiates"]  # hard round (PREREG_V2.md)
FIELD_BASE = {"str", "int", "number", "bool", "date"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
NEW_KEY = "-N"  # marker inside keys of records created by tools (hashing ignores their exact numbering)


class SpecError(Exception):
    pass


# ---------------------------------------------------------------- types and values

def parse_type(t):
    if not isinstance(t, str):
        raise SpecError(f"type must be a string, got {t!r}")
    if t in FIELD_BASE:
        return (t,)
    if t.startswith("ref:"):
        return ("ref", t[4:])
    if t.startswith("enum:"):
        vals = [v for v in t[5:].split("|")]
        if not vals or any(not v for v in vals):
            raise SpecError(f"enum needs non-empty values: {t!r}")
        return ("enum", vals)
    raise SpecError(f"unknown type {t!r} (allowed: str, int, number, bool, date, ref:<entity>, enum:a|b|...)")


def norm_date(v):
    if isinstance(v, (dt.date, dt.datetime)):
        return v.strftime("%Y-%m-%d")
    s = str(v).strip()
    for f in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(s, f).strftime("%Y-%m-%d")
        except ValueError:
            pass
    raise ValueError(f"not a date: {v!r} (use YYYY-MM-DD)")


def coerce(t, v, tables=None):
    """Normalize a value given by an agent (or a spec) to the declared type; ValueError if it does not fit."""
    k = parse_type(t)
    if v is None:
        raise ValueError("missing value")
    if k[0] == "str":
        return str(v).strip()
    if k[0] == "int":
        f = float(v)
        if f != int(f):
            raise ValueError(f"not an integer: {v!r}")
        return int(f)
    if k[0] == "number":
        return round(float(v), 2)
    if k[0] == "bool":
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
        raise ValueError(f"not a boolean: {v!r}")
    if k[0] == "date":
        return norm_date(v)
    if k[0] == "enum":
        s = str(v).strip().lower()
        for c in k[1]:
            if c.lower() == s:
                return c
        raise ValueError(f"{v!r} is not one of {k[1]}")
    if k[0] == "ref":
        s = str(v).strip()
        if tables is not None:
            ent = tables.get(k[1], {})
            if s not in ent:
                hit = [x for x in ent if x.lower() == s.lower()]
                if len(hit) == 1:
                    return hit[0]
                raise ValueError(f"no {k[1]} with id {s!r}")
        return s
    raise ValueError(t)


def same(a, b):
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) < 1e-9
    return a == b


# ---------------------------------------------------------------- expressions

def days_between(a, b):
    return (dt.date.fromisoformat(norm_date(b)) - dt.date.fromisoformat(norm_date(a))).days


def add_days(a, n):
    return (dt.date.fromisoformat(norm_date(a)) + dt.timedelta(days=int(n))).isoformat()


FUNCS = {"len": len, "min": min, "max": max, "abs": abs, "round": round, "days_between": days_between,
         "add_days": add_days, "lower": lambda s: s.lower() if isinstance(s, str) else s}
CMP = {ast.Eq: lambda a, b: same(a, b), ast.NotEq: lambda a, b: not same(a, b), ast.Lt: lambda a, b: a < b,
       ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b}
BIN = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
       ast.Div: lambda a, b: a / b, ast.Mod: lambda a, b: a % b}


class EvalError(Exception):
    pass


def parse_expr(src):
    if not isinstance(src, str) or not src.strip():
        raise SpecError(f"expression must be a non-empty string, got {src!r}")
    try:
        return ast.parse(src.strip(), mode="eval").body
    except SyntaxError as e:
        raise SpecError(f"cannot parse expression {src!r}: {e.msg}")


def ev(node, env, tables):
    """Evaluate a parsed expression. Ordering comparisons with None are False; == and != work with None."""
    T = type(node)
    if T is ast.Constant:
        return node.value
    if T is ast.Name:
        if node.id not in env:
            raise EvalError(f"unknown name {node.id!r}")
        return env[node.id]
    if T is ast.Attribute:
        base = ev(node.value, env, tables)
        if base is None:
            return None
        if not isinstance(base, dict):
            raise EvalError(f"cannot read field {node.attr!r} of a non-record")
        if node.attr not in base:
            raise EvalError(f"no field {node.attr!r}")
        return base[node.attr]
    if T is ast.BoolOp:
        if isinstance(node.op, ast.And):
            v = True
            for x in node.values:
                v = ev(x, env, tables)
                if not v:
                    return v
            return v
        v = False
        for x in node.values:
            v = ev(x, env, tables)
            if v:
                return v
        return v
    if T is ast.UnaryOp:
        v = ev(node.operand, env, tables)
        if isinstance(node.op, ast.Not):
            return not v
        if isinstance(node.op, ast.USub):
            return None if v is None else -v
        raise EvalError("unsupported unary operator")
    if T is ast.BinOp:
        a, b = ev(node.left, env, tables), ev(node.right, env, tables)
        if a is None or b is None:
            return None
        if type(node.op) not in BIN:
            raise EvalError("unsupported operator")
        try:
            return BIN[type(node.op)](a, b)
        except Exception as e:
            raise EvalError(f"arithmetic failed: {e}")
    if T is ast.Compare:
        left = ev(node.left, env, tables)
        for op, rn in zip(node.ops, node.comparators):
            right = ev(rn, env, tables)
            if isinstance(op, (ast.In, ast.NotIn)):
                if right is None:
                    r = False
                else:
                    try:
                        r = any(same(left, x) for x in right) if isinstance(right, (list, tuple)) else (left in right)
                    except TypeError as e:
                        raise EvalError(f"'in' failed: {e}")
                r = r if isinstance(op, ast.In) else not r
            elif isinstance(op, (ast.Is, ast.IsNot)):
                r = (left is right) if isinstance(op, ast.Is) else (left is not right)
            else:
                if type(op) not in CMP:
                    raise EvalError("unsupported comparison")
                if (left is None or right is None) and type(op) not in (ast.Eq, ast.NotEq):
                    r = False
                else:
                    try:
                        r = CMP[type(op)](left, right)
                    except TypeError as e:
                        raise EvalError(f"comparison failed: {e}")
            if not r:
                return False
            left = right
        return True
    if T is ast.IfExp:
        return ev(node.body, env, tables) if ev(node.test, env, tables) else ev(node.orelse, env, tables)
    if T in (ast.List, ast.Tuple):
        return [ev(x, env, tables) for x in node.elts]
    if T is ast.Call:
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise EvalError("only plain calls of allowed functions")
        fn = node.func.id
        if fn == "db_count":  # db_count("entity", "field", value): records of entity whose field equals value
            if len(node.args) != 3:
                raise EvalError("db_count takes 3 arguments")
            ent, fld = ev(node.args[0], env, tables), ev(node.args[1], env, tables)
            val = ev(node.args[2], env, tables)
            return sum(1 for r in tables.get(ent, {}).values() if same(r.get(fld), val))
        if fn not in FUNCS:
            raise EvalError(f"unknown function {fn!r}")
        args = [ev(a, env, tables) for a in node.args]
        if any(a is None for a in args):
            return None
        try:
            return FUNCS[fn](*args)
        except Exception as e:
            raise EvalError(f"{fn} failed: {e}")
    raise EvalError(f"unsupported syntax: {T.__name__}")


def static_refs(node):
    """(name, attr) pairs read in an expression, plain names, called functions, db_count entity/field constants."""
    pairs, names, calls, dbc = set(), set(), set(), set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            pairs.add((n.value.id, n.attr))
        elif isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            calls.add(n.func.id)
            if n.func.id == "db_count" and len(n.args) == 3 and all(isinstance(a, ast.Constant) for a in n.args[:2]):
                dbc.add((n.args[0].value, n.args[1].value))
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Attribute, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
               ast.USub, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Compare, ast.Eq, ast.NotEq, ast.Lt,
               ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.IfExp, ast.List, ast.Tuple, ast.Call,
               ast.Load)
    bad = sorted({type(n).__name__ for n in ast.walk(node) if not isinstance(n, allowed)})
    return pairs, names, calls, dbc, bad


# ---------------------------------------------------------------- database

def build_tables(spec, case):
    tables = {e: {} for e in spec["entities"]}
    for src in (spec.get("db_common") or {}, case.get("db") or {}):
        for ent, recs in src.items():
            if ent not in spec["entities"]:
                raise SpecError(f"db has records for unknown entity {ent!r}")
            E = spec["entities"][ent]
            for r in recs:
                if not isinstance(r, dict):
                    raise SpecError(f"{ent}: record must be an object")
                if set(r) != set(E["fields"]):
                    miss, extra = set(E["fields"]) - set(r), set(r) - set(E["fields"])
                    raise SpecError(f"{ent} record {r.get(E['key'])!r}: fields must be exactly the declared ones "
                                    f"(missing {sorted(miss)}, extra {sorted(extra)})")
                rr = {}
                for f, t in E["fields"].items():
                    if r[f] is None:
                        rr[f] = None
                        continue
                    try:
                        rr[f] = coerce(t, r[f])
                    except ValueError as e:
                        raise SpecError(f"{ent} record {r.get(E['key'])!r} field {f!r}: {e}")
                k = rr[E["key"]]
                if k in tables[ent]:
                    raise SpecError(f"{ent}: duplicate key {k!r}")
                tables[ent][k] = rr
    for ent, E in spec["entities"].items():  # foreign keys must resolve
        for k, r in tables[ent].items():
            for f, t in E["fields"].items():
                pt = parse_type(t)
                if pt[0] == "ref" and r[f] is not None and r[f] not in tables.get(pt[1], {}):
                    raise SpecError(f"{ent} {k!r}: field {f!r} points to missing {pt[1]} {r[f]!r}")
    return tables


def db_hash_view(tables):
    out = {}
    for ent, recs in sorted(tables.items()):
        fixed = {k: v for k, v in recs.items() if NEW_KEY not in k}
        created = sorted(json.dumps({f: x for f, x in v.items() if NEW_KEY not in str(x)}, sort_keys=True, default=str)
                         for k, v in recs.items() if NEW_KEY in k)
        out[ent] = {"fixed": fixed, "created": created}
    return out


# ---------------------------------------------------------------- tools

def tool_is_write(t):
    return t["kind"] in ("update", "create")


def run_tool(spec, tables, counters, t, kwargs):
    """Execute one tool on tables (mutated in place). Returns the JSON-serializable result; errors are results too."""
    E = spec["entities"][t["entity"]]
    tab = tables[t["entity"]]
    try:
        if t["kind"] == "get":
            k = coerce(f"ref:{t['entity']}", kwargs.get(E["key"]), tables)
            return copy.deepcopy(tab[k])
        if t["kind"] == "search":
            want = {}
            for f in t["by"]:
                if kwargs.get(f) is None:
                    raise ValueError(f"missing argument {f!r}")
                want[f] = coerce(E["fields"][f], kwargs[f], None)
            hits = [copy.deepcopy(r) for r in tab.values() if all(same(r.get(f), v) for f, v in want.items())]
            return {"count": len(hits), "results": hits}
        args = {}
        for a, d in t["args"].items():
            if kwargs.get(a) is None:
                raise ValueError(f"missing argument {a!r}")
            args[a] = coerce(d["type"], kwargs[a], tables)
        env = {"args": args, "today": TODAY}
        if t["kind"] == "update":
            target = next(a for a, d in t["args"].items() if d["type"] == f"ref:{t['entity']}")
            rec = tab[args[target]]
            env["rec"] = copy.deepcopy(rec)
            new = {f: ev(t["_set"][f], env, tables) for f in t["set"]}
            for f, v in new.items():
                rec[f] = None if v is None else coerce(E["fields"][f], v, tables)
            return {"status": "ok", "record": copy.deepcopy(rec)}
        counters[t["entity"]] = counters.get(t["entity"], 0) + 1
        key = f"{t['entity'][:4].upper()}{NEW_KEY}{counters[t['entity']]:03d}"
        rec = {f: None for f in E["fields"]}
        rec[E["key"]] = key
        for f in t["fields"]:
            v = ev(t["_fields"][f], env, tables)
            rec[f] = None if v is None else coerce(E["fields"][f], v, tables)
        tab[key] = rec
        return {"status": "ok", "record": copy.deepcopy(rec)}
    except (ValueError, EvalError, KeyError, TypeError) as e:
        return {"error": str(e)}


# ---------------------------------------------------------------- spec checks

def check_spec(spec, brief=None):
    """Procedure-level structural checks. Raises SpecError with a message the generator can act on."""
    for k in ("procedure_id", "domain", "organization", "title", "summary", "entities", "tools", "request", "claims",
              "bindings", "outcomes", "rules", "default_outcome", "policy_text", "db_common", "cases"):
        if k not in spec:
            raise SpecError(f"missing top-level key {k!r}")
    ents = spec["entities"]
    if not isinstance(ents, dict) or not ents:
        raise SpecError("entities must be a non-empty object")
    for en, E in ents.items():
        if not NAME_RE.match(en):
            raise SpecError(f"entity name {en!r} must be snake_case")
        if E.get("key") not in E.get("fields", {}):
            raise SpecError(f"entity {en!r}: key must be one of its fields")
        if E["fields"][E["key"]] != "str":
            raise SpecError(f"entity {en!r}: key field must have type str")
        for f, t in E["fields"].items():
            if not NAME_RE.match(f):
                raise SpecError(f"{en}.{f}: field names must be snake_case")
            pt = parse_type(t)
            if pt[0] == "ref" and pt[1] not in ents:
                raise SpecError(f"{en}.{f}: ref to unknown entity {pt[1]!r}")
    names = set()
    for t in spec["tools"]:
        n = t.get("name", "")
        if not NAME_RE.match(n) or n in names or n == "transfer_to_human_agents":
            raise SpecError(f"tool name {n!r} invalid or duplicate")
        names.add(n)
        if t.get("entity") not in ents:
            raise SpecError(f"tool {n}: unknown entity {t.get('entity')!r}")
        if not t.get("description"):
            raise SpecError(f"tool {n}: needs a description")
        E = ents[t["entity"]]
        kind = t.get("kind")
        if kind == "search":
            if not t.get("by") or any(f not in E["fields"] for f in t["by"]) or E["key"] in t["by"]:
                raise SpecError(f"tool {n}: 'by' must list non-key fields of {t['entity']}")
        elif kind in ("update", "create"):
            args = t.get("args")
            if not isinstance(args, dict) or not args:
                raise SpecError(f"tool {n}: write tools need args")
            for a, d in args.items():
                if not NAME_RE.match(a) or not isinstance(d, dict) or "type" not in d:
                    raise SpecError(f"tool {n}: arg {a!r} needs a type")
                pt = parse_type(d["type"])
                if pt[0] == "str":
                    raise SpecError(f"tool {n}: arg {a!r} is free text; write tools may only take ref/enum/int/number/bool/date")
                if pt[0] == "ref" and pt[1] not in ents:
                    raise SpecError(f"tool {n}: arg {a!r} refers to unknown entity")
            if kind == "update":
                tg = [a for a, d in args.items() if d["type"] == f"ref:{t['entity']}"]
                if len(tg) != 1:
                    raise SpecError(f"tool {n}: an update tool needs exactly one arg of type ref:{t['entity']} (the record to change)")
                if not t.get("set") or any(f not in E["fields"] or f == E["key"] for f in t["set"]):
                    raise SpecError(f"tool {n}: 'set' must map non-key fields of {t['entity']} to expressions")
                t["_set"] = {f: parse_expr(x) for f, x in t["set"].items()}
                _check_names(t["_set"].values(), {"args": set(args), "rec": set(E["fields"])}, f"tool {n} set", spec)
            else:
                if not t.get("fields") or any(f not in E["fields"] or f == E["key"] for f in t["fields"]):
                    raise SpecError(f"tool {n}: 'fields' must map non-key fields of {t['entity']} to expressions")
                t["_fields"] = {f: parse_expr(x) for f, x in t["fields"].items()}
                _check_names(t["_fields"].values(), {"args": set(args)}, f"tool {n} fields", spec)
        elif kind != "get":
            raise SpecError(f"tool {n}: kind must be get, search, update or create")
    getters = {t["entity"] for t in spec["tools"] if t["kind"] == "get"}
    for p, d in spec["request"].items():
        if not NAME_RE.match(p):
            raise SpecError(f"request param {p!r} must be snake_case")
        pt = parse_type(d.get("type"))
        if pt[0] == "ref" and pt[1] not in ents:
            raise SpecError(f"request param {p}: unknown entity")
        if not d.get("description"):
            raise SpecError(f"request param {p}: needs a description")
    for c, d in spec["claims"].items():
        if not NAME_RE.match(c) or not d.get("description"):
            raise SpecError(f"claim {c!r} needs a snake_case name and a description")
        parse_type(d.get("type"))
    bnames = []
    for i, b in enumerate(spec["bindings"]):
        if not NAME_RE.match(b.get("name", "")) or b["name"] in ("req", "claim", "today", "args", "rec") or b["name"] in bnames:
            raise SpecError(f"binding {b.get('name')!r}: invalid or duplicate name")
        if b.get("entity") not in ents:
            raise SpecError(f"binding {b['name']}: unknown entity")
        if b["entity"] not in getters:
            raise SpecError(f"binding {b['name']}: entity {b['entity']!r} needs a get tool so the agent can read it")
        m = re.match(r"^(req|[a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)$", b.get("key", ""))
        if not m:
            raise SpecError(f"binding {b['name']}: key must be 'req.<param>' or '<earlier binding>.<field>'")
        src, fld = m.groups()
        if src == "req":
            if fld not in spec["request"] or spec["request"][fld]["type"] != f"ref:{b['entity']}":
                raise SpecError(f"binding {b['name']}: req.{fld} must be a request param of type ref:{b['entity']}")
        else:
            if src not in bnames:
                raise SpecError(f"binding {b['name']}: {src!r} must be an earlier binding")
            sb = next(x for x in spec["bindings"] if x["name"] == src)
            if ents[sb["entity"]]["fields"].get(fld) != f"ref:{b['entity']}":
                raise SpecError(f"binding {b['name']}: {src}.{fld} must be a field of type ref:{b['entity']}")
        if i == 0 and src != "req":
            raise SpecError("the first binding (the record the request is about) must be keyed by a request param")
        bnames.append(b["name"])
    for p, d in spec["request"].items():
        if d["type"].startswith("ref:") and not any(b["key"] == f"req.{p}" for b in spec["bindings"]):
            raise SpecError(f"request param {p} has a ref type, so some binding must be keyed by req.{p}")
    if not isinstance(spec["outcomes"], dict) or not spec["outcomes"]:
        raise SpecError("outcomes must be a non-empty object")
    tools = {t["name"]: t for t in spec["tools"]}
    scope = {"req": set(spec["request"]), "claim": set(spec["claims"]),
             **{b["name"]: set(ents[b["entity"]]["fields"]) for b in spec["bindings"]}}
    for on, o in spec["outcomes"].items():
        if not NAME_RE.match(on) or on == "cannot_complete":
            raise SpecError(f"outcome name {on!r} invalid")
        if o.get("kind") not in ("perform", "refuse"):
            raise SpecError(f"outcome {on}: kind must be perform or refuse")
        calls = o.get("calls") or []
        if o["kind"] == "refuse" and calls:
            raise SpecError(f"outcome {on}: a refuse outcome makes no calls")
        if o["kind"] == "perform" and not calls:
            raise SpecError(f"outcome {on}: a perform outcome needs at least one write call")
        for c in calls:
            t = tools.get(c.get("tool"))
            if t is None or not tool_is_write(t):
                raise SpecError(f"outcome {on}: {c.get('tool')!r} is not a write tool")
            if set(c.get("args", {})) != set(t["args"]):
                raise SpecError(f"outcome {on}: call {t['name']} must give exactly the args {sorted(t['args'])}")
            c["_args"] = {a: parse_expr(x) for a, x in c["args"].items()}
            _check_names(c["_args"].values(), scope, f"outcome {on} call {t['name']}", spec)
    if spec["default_outcome"] not in spec["outcomes"]:
        raise SpecError("default_outcome must name an outcome")
    ids = set()
    for r in spec["rules"]:
        if not r.get("id") or r["id"] in ids:
            raise SpecError(f"rule id {r.get('id')!r} missing or duplicate")
        ids.add(r["id"])
        if r.get("outcome") not in spec["outcomes"]:
            raise SpecError(f"rule {r['id']}: unknown outcome {r.get('outcome')!r}")
        if not r.get("explain"):
            raise SpecError(f"rule {r['id']}: needs 'explain'")
        r["_when"] = parse_expr(r.get("when"))
        _check_names([r["_when"]], scope, f"rule {r['id']}", spec)
    if not spec["rules"]:
        raise SpecError("at least one rule")
    if len(spec["policy_text"]) < 300:
        raise SpecError("policy_text is too short; it must state every rule in plain language")
    if brief:
        st = brief["structure"]
        if len(spec["rules"]) != st["n_rules"]:
            raise SpecError(f"the brief asks for exactly {st['n_rules']} rules, the spec has {len(spec['rules'])}")
        main = [o for o in spec["outcomes"].values() if o["kind"] == "perform"]
        if not any(len(o["calls"]) == st["n_calls"] for o in main):
            raise SpecError(f"the brief asks for a perform outcome with exactly {st['n_calls']} write calls")
        if st["escalation"] and len(main) < 2:
            raise SpecError("the brief asks for an escalation branch: a second perform outcome (e.g. a review request) chosen by a rule")
        if st["verification"]:
            ok = any(any(n == "claim" for n, _ in static_refs(r["_when"])[0]) for r in spec["rules"])
            if not ok:
                raise SpecError("the brief asks for identity verification: a rule that compares a claim with a database field")
        if len(spec["cases"]) != len(brief["cases"]):
            raise SpecError(f"the brief asks for exactly {len(brief['cases'])} cases")
        first = spec["bindings"][0]["name"]
        others = {b["name"] for b in spec["bindings"][1:]}
        if st.get("min_hidden_rules"):
            hid = [r["id"] for r in spec["rules"] if any(n in others for n, _ in static_refs(r["_when"])[0])]
            if len(hid) < st["min_hidden_rules"]:
                raise SpecError(f"the brief asks for at least {st['min_hidden_rules']} rules that check a record other than the "
                                f"main one ({first}); found {hid}")
        if st.get("computed_arg"):
            def computed(node):
                pairs, names, calls, _, _ = static_refs(node)
                arith = any(isinstance(n, (ast.BinOp, ast.Call)) for n in ast.walk(node))
                return arith and len({n for n, _ in pairs} | (names - {"req", "claim", "today"})) >= 1 and len(pairs) >= 2
            if not any(computed(c["_args"][a]) for o in spec["outcomes"].values() for c in o["calls"] for a in c["_args"]):
                raise SpecError("the brief asks for a computed argument: some write call needs an argument computed with "
                                "arithmetic or a date function from at least two values (e.g. a prorated amount or a new date)")
        if st.get("db_count"):
            if not any(static_refs(r["_when"])[3] for r in spec["rules"]):
                raise SpecError("the brief asks for a rule that uses db_count (counting related records)")


def _check_names(nodes, scope, where, spec):
    for node in nodes:
        pairs, names, calls, dbc, bad = static_refs(node)
        if bad:
            raise SpecError(f"{where}: unsupported syntax {bad}")
        for n in names:
            if n not in scope and n != "today" and n not in FUNCS and n != "db_count":
                raise SpecError(f"{where}: unknown name {n!r} (available: {sorted(scope) + ['today']})")
        for n, a in pairs:
            if n in scope and a not in scope[n]:
                raise SpecError(f"{where}: {n}.{a} does not exist")
        for f in calls:
            if f not in FUNCS and f != "db_count":
                raise SpecError(f"{where}: function {f!r} is not allowed")
        for ent, fld in dbc:
            if ent not in spec["entities"] or fld not in spec["entities"][ent]["fields"]:
                raise SpecError(f"{where}: db_count over unknown {ent}.{fld}")
            if not any(t["kind"] == "search" and t["entity"] == ent and t["by"] == [fld] for t in spec["tools"]):
                raise SpecError(f"{where}: db_count('{ent}', '{fld}', ...) needs a search tool on {ent} by ['{fld}'] so the agent can count too")


# ---------------------------------------------------------------- the correct outcome of a case

def resolve(spec, tables, req):
    recs = {}
    for b in spec["bindings"]:
        src, fld = b["key"].split(".")
        k = req.get(fld) if src == "req" else (recs[src] or {}).get(fld)
        rec = tables[b["entity"]].get(k)
        if rec is None:
            raise SpecError(f"binding {b['name']}: no {b['entity']} with key {k!r} in this case's database")
        recs[b["name"]] = rec
    return recs


def rule_fields(spec):
    out = set()
    bn = {b["name"] for b in spec["bindings"]}
    for r in spec["rules"]:
        out |= {(n, a) for n, a in static_refs(r["_when"])[0] if n in bn}
    return out


def reach(spec, tables, req, recs, facts):
    """Which bindings the agent can obtain from the given customer facts using the tools (fixed point)."""
    known = {f["fact"]: f["value"] for f in facts}
    E = spec["entities"]
    ok, how = set(), {}

    def field_known(b, f):
        rec = recs[b["name"]]
        if f"{b['name']}.{f}" in known and same(coerce(E[b["entity"]]["fields"][f], known[f"{b['name']}.{f}"]), rec[f]):
            return True
        pt = parse_type(E[b["entity"]]["fields"][f])
        if pt[0] == "ref":
            return any(o in ok and recs[o][E[ob["entity"]]["key"]] == rec[f]
                       for ob in spec["bindings"] for o in [ob["name"]] if ob["entity"] == pt[1])
        return False

    changed = True
    while changed:
        changed = False
        for b in spec["bindings"]:
            if b["name"] in ok:
                continue
            src, fld = b["key"].split(".")
            rec = recs[b["name"]]
            got = None
            if src == "req" and f"req.{fld}" in known and same(str(known[f"req.{fld}"]), str(req[fld])):
                got = "stated"
            elif src != "req" and src in ok:
                got = "via " + src
            else:
                for t in spec["tools"]:
                    if t["kind"] != "search" or t["entity"] != b["entity"]:
                        continue
                    if not all(field_known(b, f) for f in t["by"]):
                        continue
                    cands = [r for r in tables[b["entity"]].values() if all(same(r[f], rec[f]) for f in t["by"])]
                    extra = [f for f in E[b["entity"]]["fields"] if f not in t["by"] and field_known(b, f)]
                    cands = [r for r in cands if all(same(r[f], rec[f]) for f in extra)]
                    if len(cands) == 1 and cands[0] is rec:
                        got = "search " + t["name"]
                        break
            if got:
                ok.add(b["name"]); how[b["name"]] = got; changed = True
    params_ok = all(f"req.{p}" in known or (d["type"].startswith("ref:") and any(
        b["key"] == f"req.{p}" and b["name"] in ok for b in spec["bindings"])) for p, d in spec["request"].items())
    return ok, how, params_ok and len(ok) == len(spec["bindings"])


def solve(spec, case):
    """Correct outcome of a case, computed from the spec. Raises SpecError if the case is malformed."""
    tables = build_tables(spec, case)
    req = {}
    for p, d in spec["request"].items():
        if p not in case.get("request_values", {}):
            raise SpecError(f"case {case.get('case_id')}: request_values must give {p!r}")
        try:
            req[p] = coerce(d["type"], case["request_values"][p], tables)
        except ValueError as e:
            raise SpecError(f"case {case.get('case_id')}: request_values.{p}: {e}")
    recs = resolve(spec, tables, req)
    facts = case.get("customer_facts", [])
    for f in facts:
        fa = f.get("fact", "")
        if f.get("disclose") not in ("upfront", "if_asked"):
            raise SpecError(f"case {case['case_id']}: fact {fa!r}: disclose must be upfront or if_asked")
        m = re.match(r"^([a-z][a-z0-9_]*)\.([a-z][a-z0-9_]*)$", fa)
        if not m:
            raise SpecError(f"case {case['case_id']}: fact {fa!r} must look like req.<param>, claim.<name> or <binding>.<field>")
        ns, nm = m.groups()
        try:
            if ns == "req":
                coerce(spec["request"][nm]["type"], f["value"], tables)
                if not same(str(f["value"]), str(case["request_values"][nm])) and not spec["request"][nm]["type"].startswith("ref:"):
                    raise SpecError(f"case {case['case_id']}: fact req.{nm} must equal request_values.{nm}")
            elif ns == "claim":
                coerce(spec["claims"][nm]["type"], f["value"])
            else:
                b = next(x for x in spec["bindings"] if x["name"] == ns)
                coerce(spec["entities"][b["entity"]]["fields"][nm], f["value"])
        except (KeyError, StopIteration):
            raise SpecError(f"case {case['case_id']}: fact {fa!r} names an unknown param, claim, binding or field")
        except ValueError as e:
            raise SpecError(f"case {case['case_id']}: fact {fa!r}: {e}")
    def decide(fs):
        claim = {c: None for c in spec["claims"]}
        for f in fs:
            if f["fact"].startswith("claim."):
                claim[f["fact"][6:]] = coerce(spec["claims"][f["fact"][6:]]["type"], f["value"])
        ok, how, suff = reach(spec, tables, req, recs, fs)
        d = {"sufficient": suff, "reached": how, "claim": claim}
        if not suff:
            d.update(outcome="cannot_complete", decided_by="insufficient", calls=[], cls="refuse")
            return d
        env = {"req": req, "claim": claim, "today": TODAY, **recs}
        try:
            hit = next((r for r in spec["rules"] if ev(r["_when"], env, tables)), None)
        except EvalError as e:
            raise SpecError(f"case {case['case_id']}: evaluating the rules failed: {e}")
        oname = hit["outcome"] if hit else spec["default_outcome"]
        calls = []
        for c in spec["outcomes"][oname]["calls"]:
            try:
                calls.append({"name": c["tool"], "arguments": {a: ev(x, env, tables) for a, x in c["_args"].items()}})
            except EvalError as e:
                raise SpecError(f"case {case['case_id']}: outcome {oname} call {c['tool']}: {e}")
        d.update(outcome=oname, decided_by=hit["id"] if hit else "default", calls=calls, cls=spec["outcomes"][oname]["kind"])
        return d

    out = decide(facts)
    up = decide([f for f in facts if f["disclose"] == "upfront"])
    out["req"] = req
    out["sufficient_upfront_only"] = up["sufficient"]
    # the agent must ask when the facts the customer volunteers lead somewhere else than all the facts it knows
    out["needs_asking"] = (up["outcome"], up["calls"]) != (out["outcome"], out["calls"])
    return out, tables


def gold_end_state(spec, case, sol, tables):
    """Execute the correct calls; every one must succeed and a perform outcome must change the database."""
    tools = {t["name"]: t for t in spec["tools"]}
    t2 = copy.deepcopy(tables)
    before = json.dumps(db_hash_view(t2), sort_keys=True, default=str)
    for c in sol["calls"]:
        r = run_tool(spec, t2, {}, tools[c["name"]], c["arguments"])
        if "error" in r:
            raise SpecError(f"case {case['case_id']}: the correct call {c['name']}({c['arguments']}) fails: {r['error']}")
    if sol["cls"] == "perform" and json.dumps(db_hash_view(t2), sort_keys=True, default=str) == before:
        raise SpecError(f"case {case['case_id']}: the correct calls of outcome {sol['outcome']} do not change the database")
    return t2


def case_features(spec, case, sol, tables):
    """Measured properties of a case, independent of what the generator declared."""
    facts = case.get("customer_facts", [])
    rf = rule_fields(spec)
    recs = resolve(spec, tables, sol["req"])
    first = spec["bindings"][0]
    p0 = first["key"].split(".")[1]
    ent_of = {b["name"]: b["entity"] for b in spec["bindings"]}
    conflicts, correct_rule_facts = [], []
    for f in facts:
        ns, nm = f["fact"].split(".")
        if ns not in ent_of or (ns, nm) not in rf:
            continue
        v = coerce(spec["entities"][ent_of[ns]]["fields"][nm], f["value"])
        (correct_rule_facts if same(v, recs[ns][nm]) else conflicts).append(f["fact"])
    ex = case.get("extra_details", [])
    return {"sufficient": sol["sufficient"], "needs_asking": sol["needs_asking"],
            "primary_lookup": f"req.{p0}" not in {f["fact"] for f in facts} and sol["sufficient"],
            "n_facts": len(facts), "n_if_asked": sum(f["disclose"] == "if_asked" for f in facts),
            "n_helpful": sum(e.get("kind") == "helpful" for e in ex), "n_distractor": sum(e.get("kind") == "distractor" for e in ex),
            "n_conflict": len(conflicts), "n_correct_rule_facts_upfront": sum(1 for f in facts if f["fact"] in correct_rule_facts and f["disclose"] == "upfront"),
            "n_primary_records": len([r for r in (case.get("db") or {}).get(first["entity"], [])]),
            "n_claims_given": sum(f["fact"].startswith("claim.") for f in facts),
            "changes_mind": bool(case.get("initial_request_values")), "cls": sol["cls"], "outcome": sol["outcome"],
            "decided_by": sol["decided_by"], "n_calls": len(sol["calls"]),
            "rule_index": next((i for i, r in enumerate(spec["rules"]) if r["id"] == sol["decided_by"]), len(spec["rules"]))}


def check_case_plan(spec, case, plan, sol, feat):
    """Does the case realize the planned information level, behaviour and class? Returns a list of problems."""
    p, lev = [], plan["level"]
    facts = case.get("customer_facts", [])
    up_only = all(f["disclose"] == "upfront" for f in facts)
    if plan["cls"] != sol["cls"]:
        p.append(f"planned class {plan['cls']} but the rules give outcome {sol['outcome']} ({sol['cls']})")
    if case.get("expected_outcome") != sol["outcome"]:
        p.append(f"expected_outcome says {case.get('expected_outcome')!r} but the rules give {sol['outcome']!r} "
                 f"(decided by {sol['decided_by']})")
    if lev != "insufficient" and not sol["sufficient"]:
        p.append(f"level {lev} needs a case the agent can complete, but the customer facts do not let the agent identify "
                 f"every record (reached only {sorted(sol['reached'])}) or give every request param")
    if lev == "insufficient" and sol["sufficient"]:
        p.append("level insufficient needs a case where the customer facts and tools do NOT let the agent identify the records")
    if lev in ("upfront", "helpful_extra", "distracting", "conflicting", "on_request") and feat["primary_lookup"]:
        p.append(f"level {lev}: the customer must know the id of the main record (fact req.{spec['bindings'][0]['key'].split('.')[1]})")
    if lev == "upfront" and (not up_only or feat["n_distractor"] or feat["n_conflict"] or feat["n_helpful"]):
        p.append("level upfront: every fact disclose=upfront, no extra details, no conflicting facts")
    if lev == "on_request" and not feat["needs_asking"]:
        p.append("level on_request: the agent must be unable to finish without asking (some needed fact disclose=if_asked)")
    if lev == "lookup" and not feat["primary_lookup"]:
        p.append("level lookup: the customer does not know the main record's id; the agent must find it with a search tool")
    if lev == "helpful_extra" and (feat["n_helpful"] < 2 or feat["n_correct_rule_facts_upfront"] < 1):
        p.append("level helpful_extra: at least 2 helpful extra details and at least one correct upfront fact about a field the rules check")
    if lev == "distracting" and (feat["n_distractor"] < 3 or feat["n_primary_records"] < 2):
        p.append(f"level distracting: at least 3 distractor details and at least 2 {spec['bindings'][0]['entity']} records in the case database")
    if lev == "conflicting" and feat["n_conflict"] < 1:
        p.append("level conflicting: at least one upfront fact about a field the rules check, whose value differs from the database")
    if lev != "conflicting" and feat["n_conflict"]:
        p.append(f"level {lev}: no conflicting facts allowed")
    if (plan["behavior"] in ("changes_mind", "negotiates")) != bool(case.get("initial_request_values")):
        p.append("behaviours changes_mind and negotiates need initial_request_values (what the customer asks first); other behaviours must not have it")
    if plan["behavior"] == "negotiates" and case.get("initial_request_values"):
        try:
            c0 = dict(case, request_values={**case["request_values"], **case["initial_request_values"]},
                      customer_facts=[dict(f, value=case["initial_request_values"][f["fact"][4:]]) if f["fact"][4:] in case["initial_request_values"] and f["fact"].startswith("req.") else f
                                      for f in facts])
            s0, _ = solve(spec, c0)
            if s0["cls"] != "refuse" or s0["decided_by"] == "insufficient":
                p.append(f"behaviour negotiates: the first request (initial_request_values) must be refused by a rule, but the rules give {s0['outcome']}")
        except SpecError as e:
            p.append(f"behaviour negotiates: the first request cannot be evaluated: {e}")
    if case.get("initial_request_values"):
        iv = case["initial_request_values"]
        if set(iv) - set(spec["request"]) or all(same(str(iv[k]), str(case["request_values"][k])) for k in iv):
            p.append("initial_request_values must use request param names and differ from request_values in at least one")
    for k in ("persona", "reason_for_call", "task_instructions"):
        if not case.get(k):
            p.append(f"missing {k}")
    txt = " ".join(str(case.get(k, "")) for k in ("persona", "reason_for_call", "task_instructions"))
    ids = set()
    for src in (spec.get("db_common") or {}, case.get("db") or {}):
        for ent, recs in src.items():
            key = spec["entities"][ent]["key"]
            ids |= {str(r[key]) for r in recs if r.get(key)}
    leaked = sorted(i for i in ids if len(i) >= 3 and re.search(r"(?<![A-Za-z0-9])" + re.escape(i) + r"(?![A-Za-z0-9])", txt))
    if leaked:
        p.append(f"persona/reason_for_call/task_instructions must not contain record ids ({leaked}); ids reach the customer only through customer_facts")
    return p


# ---------------------------------------------------------------- user simulator text

def label(spec, fact):
    ns, nm = fact.split(".")
    if ns == "req":
        return spec["request"][nm]["description"]
    if ns == "claim":
        return spec["claims"][nm]["description"]
    b = next(x for x in spec["bindings"] if x["name"] == ns)
    return f"the {nm.replace('_', ' ')} on your {b['entity'].replace('_', ' ')} record"


def render_user(spec, case, plan):
    facts = case.get("customer_facts", [])
    up = [f for f in facts if f["disclose"] == "upfront"]
    ask = [f for f in facts if f["disclose"] == "if_asked"]
    known = []
    if up:
        known.append("Facts you state on your own early in the conversation:")
        known += [f"- {label(spec, f['fact'])}: {f['value']}" for f in up]
    if ask:
        known.append("Facts you give only when the agent asks for them (do not volunteer them):")
        known += [f"- {label(spec, f['fact'])}: {f['value']}" for f in ask]
    ex = case.get("extra_details", [])
    if ex:
        known.append("Other things you mention along the way, in your own words:")
        known += [f"- {e['text']}" for e in ex]
    iv = case.get("initial_request_values")
    if iv and plan is not None and plan.get("behavior") == "negotiates":
        known.append("At first you ask for: " + "; ".join(f"{spec['request'][k]['description']}: {v}" for k, v in iv.items())
                     + ". Only if the agent refuses that, you ask for the values listed above instead.")
    elif iv:
        known.append("At first you ask for: " + "; ".join(f"{spec['request'][k]['description']}: {v}" for k, v in iv.items())
                     + ". Before the agent completes anything, you change your mind and want the values listed above instead.")
    kf = {f["fact"] for f in facts}
    unk = [("the id or number of " if spec["request"][p]["type"].startswith("ref:") else "") + spec["request"][p]["description"]
           for p in spec["request"] if f"req.{p}" not in kf]
    unk += [spec["claims"][c]["description"] for c in spec["claims"] if f"claim.{c}" not in kf]
    for b in spec["bindings"]:  # identifying details the agent could search by, which this customer does not have
        for t in spec["tools"]:
            if t["kind"] == "search" and t["entity"] == b["entity"]:
                for f in t["by"]:
                    lab = label(spec, f"{b['name']}.{f}")
                    if f"{b['name']}.{f}" not in kf and not parse_type(spec["entities"][b["entity"]]["fields"][f])[0] == "ref" and lab not in unk:
                        unk.append(lab)
    instr = case["task_instructions"].strip()
    instr += ("\nIf the agent asks you to confirm an action that matches what you want, confirm it. If the agent asks for "
              "something you do not know, say you do not know it. Never invent ids, numbers or dates. When your request has been "
              "handled, or the agent has clearly told you it cannot be done and you have nothing left to ask, end the "
              "conversation with ###STOP###.")
    return {"persona": case["persona"].strip(),
            "instructions": {"domain": spec["domain"], "reason_for_call": case["reason_for_call"].strip(),
                             "known_info": "\n".join(known) if known else None,
                             "unknown_info": ("You do not know: " + "; ".join(unk) + ". If asked, say you do not know.") if unk else None,
                             "task_instructions": instr}}


def to_task(spec, case, plan, sol, gen, uid):
    return {"id": uid,
            "description": {"purpose": spec["summary"],
                            "notes": f"proc={spec['procedure_id']}; gen={gen}; level={plan['level']}; behavior={plan['behavior']}; "
                                     f"outcome={sol['outcome']}; decided_by={sol['decided_by']}"},
            "user_scenario": render_user(spec, case, plan),
            "evaluation_criteria": {"actions": [{"action_id": f"{uid}_{i}", "name": c["name"], "arguments": c["arguments"]}
                                                for i, c in enumerate(sol["calls"])],
                                    "reward_basis": ["DB"]}}


# ---------------------------------------------------------------- full validation of one generated spec

def validate(spec, brief):
    """Returns (report, errors). Procedure-level errors make the whole spec invalid; case errors drop that case."""
    errs, rep = [], {"cases": {}}
    try:
        check_spec(spec, brief)
    except SpecError as e:
        return None, [f"procedure: {e}"]
    plans = {c["case_id"]: c for c in brief["cases"]}
    seen = set()
    for case in spec["cases"]:
        cid = case.get("case_id")
        if cid not in plans or cid in seen:
            errs.append(f"case {cid!r}: case_id must be one of {sorted(plans)} and appear once")
            continue
        seen.add(cid)
        try:
            sol, tables = solve(spec, case)
            gold_end_state(spec, case, sol, tables)
            feat = case_features(spec, case, sol, tables)
            probs = check_case_plan(spec, case, plans[cid], sol, feat)
        except SpecError as e:
            probs, sol, feat = [str(e)], None, None
        except Exception as e:  # anything unexpected in a generated case is that case's problem
            probs, sol, feat = [f"case {cid}: {type(e).__name__}: {e}"], None, None
        rep["cases"][cid] = {"ok": not probs, "problems": probs, "features": feat,
                             "solution": None if sol is None else {k: sol[k] for k in ("outcome", "decided_by", "calls", "cls", "sufficient")}}
        errs += [f"case {cid}: {x}" if not x.startswith("case ") else x for x in probs]
    for cid in plans:
        if cid not in seen:
            errs.append(f"case {cid}: missing")
    return rep, errs


def toolkit_signature(t, spec):
    E = spec["entities"][t["entity"]]
    py = {"str": str, "int": int, "number": float, "bool": bool, "date": str}
    if t["kind"] == "get":
        params = [(E["key"], str, f"The {t['entity'].replace('_', ' ')} id.")]
    elif t["kind"] == "search":
        params = [(f, py.get(parse_type(E["fields"][f])[0], str), f"{f.replace('_', ' ')} to match"
                   + (" (YYYY-MM-DD)" if E["fields"][f] == "date" else "")) for f in t["by"]]
    else:
        params = []
        for a, d in t["args"].items():
            pt = parse_type(d["type"])
            desc = d.get("description", a)
            if pt[0] == "enum":
                desc += f" One of: {', '.join(pt[1])}."
            if pt[0] == "date":
                desc += " Format YYYY-MM-DD."
            params.append((a, py.get(pt[0], str), desc))
    return params


def make_env_constructor(spec, case_tables, domain_name):
    """tau2 environment constructor for one case: fresh copy of the case database on every call."""
    from tau2.environment.db import DB
    from tau2.environment.environment import Environment
    from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
    from tau2.utils import get_dict_hash
    from pydantic import Field
    from typing import Dict

    class SynthDB(DB):
        tables: Dict[str, Any] = Field(default_factory=dict)
        counters: Dict[str, int] = Field(default_factory=dict)

        def get_hash(self) -> str:
            return get_dict_hash(db_hash_view(self.tables))

    attrs = {"db": SynthDB, "__doc__": f"Tools of {spec['organization']}."}

    def make(t):
        params = toolkit_signature(t, spec)

        def method(self, **kwargs):
            return json.dumps(run_tool(spec, self.db.tables, self.db.counters, t, kwargs), default=str)

        method.__name__ = t["name"]
        doc = t["description"].strip() + "\n\nArgs:\n" + "".join(f"    {n}: {d}\n" for n, _, d in params)
        method.__doc__ = doc
        method.__signature__ = inspect.Signature(
            [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            + [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY, annotation=ty) for n, ty, _ in params], return_annotation=str)
        method.__annotations__ = {**{n: ty for n, ty, _ in params}, "return": str}
        return is_tool(ToolType.WRITE if tool_is_write(t) else ToolType.READ)(method)

    for t in spec["tools"]:
        attrs[t["name"]] = make(t)

    def transfer_to_human_agents(self, summary: str) -> str:
        """Transfer the customer to a human agent, with a summary of the issue. Only transfer when the policy says so,
        when the customer explicitly asks for a human, or when you cannot handle the request with the policy and tools.

        Args:
            summary: A summary of the customer's issue.
        """
        return "Transfer successful"

    attrs["transfer_to_human_agents"] = is_tool(ToolType.GENERIC)(transfer_to_human_agents)
    Tools = type("SynthTools", (ToolKitBase,), attrs)
    policy = f"# {spec['organization']}: {spec['title']}\n\nToday is {TODAY}.\n\n{spec['policy_text'].strip()}"

    def constructor(solo_mode: bool = False, **kw):
        return Environment(domain_name=domain_name, policy=policy, tools=Tools(SynthDB(tables=copy.deepcopy(case_tables))))

    return constructor
