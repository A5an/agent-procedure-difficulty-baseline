import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys, pandas as pd
import harness as H
import pertype_models as PM
from pathlib import Path
OUT = Path(ZR.SCREEN + "/pertype/run")
OUT.mkdir(exist_ok=True)
R = H.REPO / "data/sopbench/features"
ty = pd.read_csv("types.csv")
oh = pd.concat([ty[["case_id"]], pd.get_dummies(ty[["fam", "tree", "dom"]]).astype(float)], axis=1)
oh.to_csv(OUT / "onehot.csv", index=False)
ad = pd.read_csv(R / "adele.csv").rename(columns={"task_id": "case_id"})
ad2 = ad.merge(oh, on="case_id"); ad2.to_csv(OUT / "adele_onehot.csv", index=False)
schemes = sys.argv[1:] or ["new_procedures", "new_domain"]
H.run_all(OUT, PM.make_preds, extra_csv={"onehot": {"sopbench": OUT / "onehot.csv"}, "adele_onehot": {"sopbench": OUT / "adele_onehot.csv"}},
          benches=["sopbench"], schemes=schemes)
