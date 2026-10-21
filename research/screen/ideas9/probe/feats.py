"""Build feature tables from states/*.npz (no labels). usage: feats.py. Transductive unsupervised PCA, see PREREG.md."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import json, numpy as np, pandas as pd
from pathlib import Path
R = Path(ZR.REPO + "/data")
HERE = Path(ZR.SCREEN + "/ideas9/probe"); ST = HERE / "states"
LAYERS = [12, 16, 20, 24, 28, 32]; LI = LAYERS.index(24); NPC = 16
def load(bench):
    tasks = pd.read_csv(R / bench / "tasks.csv"); ids = list(tasks.case_id)
    Z = [np.load(ST / (bench + "__" + c.replace("/", "__") + ".npz")) for c in ids]
    last = np.stack([z["last"].astype(np.float32) for z in Z]); ref = np.stack([z["ref"].astype(np.float32) for z in Z])
    mreq = np.stack([z["mreq"].astype(np.float32) for z in Z]); sc = np.stack([z["sc"] for z in Z])
    return tasks, ids, last, ref, mreq, sc
def pca(X, k=NPC):
    Xc = X - X.mean(0); U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T, Vt, s
def group_center(X, g):
    X = X.copy()
    for v in np.unique(g): X[g == v] -= X[g == v].mean(0)
    return X
def group_of(bench, tasks): return tasks.procedure_id.values if bench == "sopbench" else tasks.domain.values
if __name__ == "__main__":
    for bench in ["sopbench", "tau2", "tauk_banking"]:
        tasks, ids, last, ref, mreq, sc = load(bench)
        X = last[:, LI]
        pcs, _, _ = pca(X); pcc, _, _ = pca(group_center(X, group_of(bench, tasks)))
        cols = [f"pc{i+1}" for i in range(NPC)]
        pd.DataFrame(pcs, index=ids, columns=cols).rename_axis("case_id").to_csv(HERE / f"probe_{bench}.csv")
        pd.DataFrame(pcc, index=ids, columns=cols).rename_axis("case_id").to_csv(HERE / f"probec_{bench}.csv")
        shift = np.linalg.norm(last - ref, axis=2)
        d = {f"shift{L}": shift[:, i] for i, L in enumerate(LAYERS)}
        d["nll_cond"] = sc[:, 0]; d["pmi"] = sc[:, 1] - sc[:, 0]
        pd.DataFrame(d, index=ids).rename_axis("case_id").to_csv(HERE / f"scal_{bench}.csv")
        pd.DataFrame({"log_nreq": np.log(sc[:, 2]), "log_nA": np.log(sc[:, 3])}, index=ids).rename_axis("case_id").to_csv(HERE / f"len_{bench}.csv")
        print(bench, len(ids), X.shape)
