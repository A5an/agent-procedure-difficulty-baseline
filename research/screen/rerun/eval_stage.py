import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import sys; sys.path.insert(0,ZR.SCREEN)
import harness as H
d=sys.argv[1]
new={"probe_L18":"hidden state L18 ridge","probe_L36":"hidden state L36 ridge","entropy":"token entropy ridge","llm_forecast":"PRECOG-style forecast ridge","emb_bge":"bge-base ridge","bpm_complexity":"BPM complexity (sopbench)","amortized_irt_bge":"amortized IRT on bge"}
H.evaluate_all(d,new)
