"""Extract per-attempt outcomes of tau2 (trial index -> success) so that 'first attempt only' is available."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, json, re, collections
sys.path.insert(0, ZR.SCREEN + "/ideas")
from pathlib import Path
from cpulock import cpu_lock
T = Path(ZR.DATA + "/tau2-domains/trajectories")
R = Path(ZR.REPO + "/data/tau2")
idmap = json.loads((R / "id_map.json").read_text())
out = collections.defaultdict(dict)
with cpu_lock("pilot"):
    for dom in ["airline", "retail", "telecom"]:
        for f in sorted(T.glob("*.json")):
            if not re.search(rf"(^|[_-]){dom}([_.-]|$)", f.name):
                continue
            d = json.loads(f.read_text())
            subj = f.name.split("__")[0]
            for s in d.get("simulations") or d.get("results") or []:
                tid = s.get("task_id"); trial = s.get("trial", s.get("trial_id", 0))
                rw = (s.get("reward_info") or {}).get("reward", s.get("reward"))
                if tid is None or rw is None: continue
                key = idmap.get(f"{dom}:{tid}")
                if key is None: continue
                out[subj].setdefault(key, {})[int(trial)] = float(rw) >= 0.999
            print(f.name[:40], flush=True)
json.dump(out, open("tau2_attempts.json", "w"))
# check against responses
bad = 0
for line in open(R / "responses.jsonl"):
    r = json.loads(line)
    for k, v in r["responses"].items():
        a = out[r["subject_id"]].get(k, {})
        if sum(a.values()) != v["successes"] or len(a) != v["trials"]: bad += 1
print("mismatch cells", bad)
