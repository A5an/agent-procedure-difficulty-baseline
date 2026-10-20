"""Step 1: tool table. Schema-derived columns by code, ID requires/produces by a small hand table. Output tool_table.json."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re
from pathlib import Path
I3 = Path(ZR.SCREEN + "/ideas3/plan_tools")
raw = json.load(open(I3 / "tools_raw.json")); types = json.load(open(I3 / "tool_types.json"))
# ID types the customer can state from memory (hand judgement, fixed before any result): depth 0 roots
STATABLE = {"airline": {"user_id", "reservation_id", "airport", "date", "passenger_info", "cabin", "text", "number"},
            "retail": {"order_id", "name_zip", "email", "text", "address", "number"},
            "telecom": {"phone", "full_name_dob", "text", "number"}}
# argument name -> ID type (hand table, per domain)
ARG = {"airline": {"user_id": "user_id", "reservation_id": "reservation_id", "origin": "airport", "destination": "airport", "date": "date",
                   "flight_number": "flight_number", "payment_id": "payment_method_id", "payment_methods": "payment_method_id",
                   "flights": "flight_number", "passengers": "passenger_info", "cabin": "cabin"},
       "retail": {"order_id": "order_id", "user_id": "user_id", "item_id": "item_id", "item_ids": "item_id", "new_item_ids": "item_id",
                  "product_id": "product_id", "payment_method_id": "payment_method_id", "first_name": "name_zip", "last_name": "name_zip",
                  "zip": "name_zip", "email": "email", "address1": "address", "address2": "address", "city": "address", "state": "address", "country": "address"},
       "telecom": {"phone_number": "phone", "customer_id": "customer_id", "full_name": "full_name_dob", "dob": "full_name_dob",
                   "line_id": "line_id", "bill_id": "bill_id", "id": "any_id"}}
# what a read tool yields (hand table)
PRODUCES = {"airline/get_user_details": ["reservation_id", "payment_method_id", "passenger_info"],
            "airline/get_reservation_details": ["flight_number", "payment_method_id", "passenger_info", "user_id"],
            "airline/search_direct_flight": ["flight_number"], "airline/search_onestop_flight": ["flight_number"],
            "airline/list_all_airports": ["airport"], "airline/get_flight_status": [],
            "retail/find_user_id_by_name_zip": ["user_id"], "retail/find_user_id_by_email": ["user_id"],
            "retail/get_user_details": ["order_id", "payment_method_id"], "retail/get_order_details": ["item_id", "product_id", "payment_method_id"],
            "retail/get_product_details": ["item_id"], "retail/get_item_details": [], "retail/list_all_product_types": ["product_id"],
            "telecom/get_customer_by_phone": ["customer_id", "line_id", "bill_id"], "telecom/get_customer_by_id": ["line_id", "bill_id"],
            "telecom/get_customer_by_name": ["customer_id", "line_id", "bill_id"], "telecom/get_details_by_id": ["line_id", "bill_id"],
            "telecom/get_bills_for_customer": ["bill_id"], "telecom/get_data_usage": []}
CROSS = re.compile(r"\b(must|only|same|match|matches|should|unless|cannot|can't|different|at least|at most|exactly|equal|both|either|otherwise|total|limit|if|before|after|not)\b", re.I)
FMT = re.compile(r"\b(format|such as|e\.g\.|yyyy|one of|must be|should be|options)\b", re.I)

def resolve(s, defs):
    while "$ref" in s: s = defs[s["$ref"].split("/")[-1]]
    if "anyOf" in s:  # optional types
        for a in s["anyOf"]:
            if a.get("type") != "null": return resolve(a, defs)
    return s
def depth(s, defs, d=0, seen=()):
    s = resolve(s, defs)
    if s.get("type") == "array" and "items" in s: return depth(s["items"], defs, d + 1, seen)
    if s.get("type") == "object" and "properties" in s:
        return max([depth(v, defs, d + 1, seen) for v in s["properties"].values()] or [d + 1])
    return d
def walk(s, defs):
    """yield every (leaf schema) with description for enum/format/cross counting"""
    s0 = s; s = resolve(s, defs)
    yield s0, s
    if s.get("type") == "array" and "items" in s: yield from walk(s["items"], defs)
    if s.get("type") == "object": 
        for v in (s.get("properties") or {}).values(): yield from walk(v, defs)

table = {}
for dom in raw:
    for tl in raw[dom]["agent"]:
        f = tl["function"]; name = f["name"]; key = f"{dom}/{name}"; P = f["parameters"]; defs = P.get("$defs", {})
        props = P.get("properties", {}); req = P.get("required") or []
        n_list_obj = n_enum = n_fmt = 0; cross = len(CROSS.findall(f.get("description", "")))
        for a, sch in props.items():
            r = resolve(sch, defs)
            if r.get("type") == "array" and resolve(r.get("items", {}), defs).get("type") == "object": n_list_obj += 1
            for s0, s in walk(sch, defs):
                d = (s0.get("description") or "") + " " + (s.get("description") or "")
                if "enum" in s or "enum" in s0 or "pattern" in s or "format" in s: n_enum += 1
                elif FMT.search(d): n_fmt += 1
                cross += len(CROSS.findall(s0.get("description") or ""))
        ids = {a: ARG[dom][a] for a in req if a in ARG[dom]}
        table[key] = dict(domain=dom, type=types.get(key, "generic"), req_args=req, n_args=len(props), n_req=len(req),
                          requires_ids=ids, produces=PRODUCES.get(key, []), n_list_obj=n_list_obj,
                          nest_depth=max([depth(v, defs) for v in props.values()] or [0]), n_enum_fmt=n_enum + n_fmt,
                          cross_words=cross, n_unstatable=sum(1 for t in set(ids.values()) if t not in STATABLE[dom]))
json.dump(dict(tools=table, statable={k: sorted(v) for k, v in STATABLE.items()}), open("tool_table.json", "w"), indent=1)
for k, v in table.items(): print(k, v["type"], dict(ids=v["requires_ids"], prod=v["produces"], lo=v["n_list_obj"], nd=v["nest_depth"], ef=v["n_enum_fmt"], cw=v["cross_words"], un=v["n_unstatable"]))
