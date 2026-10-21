"""Harness self-test: the baseline alone must reproduce the repository number (pooled rho 0.131)."""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import harness as H
out = ZR.SCREEN + "/_selftest"
H.run_all(out, lambda src, M: {"length": H.ridge(src["length"])}, benches=["sopbench", "tau2"], schemes=["new_procedures", "new_domain"])
r = H.evaluate_all(out, {"length": "text length"})
print("BASELINE", r["pooled_new_process"]["adele"]["rho_mean"], "LENGTH", r["pooled_new_process"]["length"]["rho_mean"])
