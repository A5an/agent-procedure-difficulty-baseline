"""Gold action lists from tau2 task files. VALIDATION AND ORACLE ONLY, never a feature of the L0 methods."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, re
from pathlib import Path
import pandas as pd
REPO = Path(ZR.REPO)
TD = Path(ZR.DATA + "/tau2-domains/tasks")
HERE = Path(__file__).parent
WRITE = re.compile(r"^(book_|cancel_|update_|modify_|exchange_|return_|send_|transfer_|change_|enable_|disable_|toggle_|set_|reset_|reboot|refuel|resume|connect|turn_|grant|make_payment|pay_|reseat|unseat|reset)")
def build():
    m = json.load(open(REPO / "data/tau2/id_map.json"))
    rows = []
    for dom in ["airline", "retail", "telecom"]:
        tk = {str(t["id"]): t for t in json.load(open(TD / f"{dom}_tasks.json"))}
        for k, c in m.items():
            if not k.startswith(dom + ":"): continue
            acts = (tk[k.split(":", 1)[1]]["evaluation_criteria"].get("actions") or [])
            names = [a["name"] for a in acts]
            w = [n for n in names if WRITE.match(n)]
            rows.append(dict(case_id=c, gold_len=len(names), gold_writes=len(w), gold_distinct=len(set(names)),
                             gold_agent_len=sum(a.get("requestor", "assistant") != "user" for a in acts)))
    pd.DataFrame(rows).to_csv(HERE / "gold_tau2.csv", index=False)
    return pd.DataFrame(rows)
if __name__ == "__main__":
    d = build(); print(d.groupby(d.case_id.str.split("/").str[0]).describe().T.to_string()[:3000])
