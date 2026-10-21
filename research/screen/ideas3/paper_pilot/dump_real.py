import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, os, json
sys.argv = ["analyze.py"]
H = ZR.SCREEN + "/ideas2/harness"
sys.path.insert(0, H); os.chdir(H)
import analyze as A
out = {}
for m in ["gemini-2.5-flash", "gemini-3.8-flash"]:
    d = A.load_arm(m, "H0", strip=False); res = {}
    for l in open(f"{A.R.OUT}/{m}__H0.jsonl"):
        r = json.loads(l)
        if r.get("error") or "interaction" not in r: continue
        cid = r["case_id"]
        tools = [tc["function"]["name"] for msg in r["interaction"] for tc in (msg.get("tool_calls") or [])]
        res[cid] = {"success": int(d[cid]["success"]), "performed": int(d[cid]["performed"]), "tools": tools, "should": int(d[cid]["should"])}
    out[m] = res
json.dump(out, open(ZR.SCREEN + "/ideas3/paper_pilot/real_h0.json", "w"))
print({m: len(v) for m, v in out.items()}, {m: sum(x["success"] for x in v.values()) / len(v) for m, v in out.items()})
