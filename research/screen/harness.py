"""Shared harness for screening new methods under exactly the baseline protocol.

Every method is run on the same folds as the baseline (fold IRT read from the repository cache),
together with the baseline itself, and scored by the repository's evaluate.py (same metrics, same
bootstrap, paired difference to the baseline). Nothing in the public repository is written.

Run any script that uses this harness with the repository interpreter and a fixed hash seed:
  PYTHONHASHSEED=0 <repo>/.venv/bin/python my_script.py

Usage inside a script:
  import harness as H
  def make_preds(src, M):            # M is the repository's `methods` module, src the feature sources
      return {"my_method": H.ridge(src["my_feature"])}
  H.run_all(out_dir, make_preds, extra_csv={"my_feature": {"sopbench": path, "tau2": path, ...}})
  H.evaluate_all(out_dir, {"my_method": "description"}, combos=[("my_method", "adele")])

Feature CSV format: first column case_id, then numeric columns (one row per task of the benchmark).
"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402

import json
import os
import sys
from pathlib import Path

assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"
REPO = Path(ZR.REPO)
sys.path.insert(0, str(REPO / "src"))
_cwd = os.getcwd()
import run_baseline as RB  # noqa: E402  (changes cwd to the agent-psychometrics checkout)
import evaluate as E  # noqa: E402
import methods as M  # noqa: E402
os.chdir(_cwd)

import pandas as pd  # noqa: E402

BENCHES = ["sopbench", "tau2", "tauk_banking"]
SCHEMES = ["random_cases", "new_procedures", "new_domain"]
CSVFeatureSource = RB.CSVFeatureSource
EmbeddingFeatureSource = RB.EmbeddingFeatureSource
GroupedFeatureSource = RB.GroupedFeatureSource


def ridge(source):
    """The baseline's model (their FeatureBasedPredictor with their alpha grid) on any feature source."""
    return RB.DifficultyPredictorAdapter(RB.FeatureBasedPredictor(source, alphas=RB.ALPHAS))


def grouped(sources, grids):
    """Their grouped ridge: one penalty grid per feature family. grids: {source_name: list of alphas}."""
    return RB.DifficultyPredictorAdapter(RB.GroupedRidgePredictor(GroupedFeatureSource(sources), alpha_grids=grids))


def run_all(out_dir, make_preds, extra_csv=None, extra_sources=None, benches=BENCHES, schemes=SCHEMES):
    """Write out_dir/<bench>/obs_<scheme>.csv.gz with constant, oracle, the baseline 'adele' and the new methods."""
    out_dir = Path(out_dir)
    base_preds = RB.predictors
    for bench in benches:
        tasks = pd.read_csv(RB.DATA / bench / "tasks.csv")
        sch = [s for s in schemes if not (s == "new_domain" and tasks.domain.nunique() < 2)
               and not (s == "new_procedures" and tasks.procedure_id.nunique() == len(tasks))]
        src = RB.sources(bench)
        for name, paths in (extra_csv or {}).items():
            if bench in paths:
                src[name] = CSVFeatureSource(Path(paths[bench]), name=name)
        for name, per_bench in (extra_sources or {}).items():
            if bench in per_bench:
                src[name] = per_bench[bench]

        def preds(s, bench=bench):
            b = base_preds(s)
            p = {"constant": b["constant"], "oracle": b["oracle"], "adele": b["adele"]}
            p.update(make_preds(s, M) if make_preds.__code__.co_argcount == 2 else make_preds(s, M, bench))
            return p
        RB.predictors = preds
        (out_dir / bench).mkdir(parents=True, exist_ok=True)
        for scheme in sch:
            obs, betas = RB.run_scheme(bench, tasks, scheme, src)
            obs.to_csv(out_dir / bench / f"obs_{scheme}.csv.gz", index=False)
            betas.to_csv(out_dir / bench / f"betas_{scheme}.csv", index=False)
    RB.predictors = base_preds


def evaluate_all(out_dir, new_methods, combos=()):
    """Score with the repository evaluator. new_methods: {column: description}. Writes eval.json and eval.txt."""
    out_dir = Path(out_dir)
    E.RESULTS = out_dir
    E.METHODS = {"constant": "agent ability only", "adele": "BASELINE", **new_methods, "oracle": "oracle"}
    E.COMBOS = list(combos)
    for combo in combos:  # equal-weight logit averages are formed by the repository's load()
        E.METHODS["+".join(combo)] = "logit average of " + ", ".join(combo)
    out, draws, lines = {}, {}, []
    for d in BENCHES:
        for s in SCHEMES:
            if (out_dir / d / f"obs_{s}.csv.gz").exists():
                out[f"{d}|{s}"], draws[f"{d}|{s}"] = E.evaluate(d, s)
                lines.append(f"{d} {s}")
                for c, r in out[f"{d}|{s}"].items():
                    lines.append(f"  {c:34s} " + " ".join(
                        f"{k}={v:+.3f}" if isinstance(v, float) else f"{k}={[round(x, 3) for x in v]}" for k, v in r.items()))
    if all(k in draws for k in E.NEW):
        out["pooled_new_process"] = E.pooled(out, draws)
        lines.append("pooled over the four new-process scenarios")
        for c, r in sorted(out["pooled_new_process"].items(), key=lambda x: -x[1]["rho_mean"]):
            lines.append(f"  {c:34s} rho {r['rho_mean']:+.3f} {[round(x, 3) for x in r['rho_ci']]}  worst {r['rho_worst']:+.3f}  "
                         f"by scenario {[round(x, 3) for x in r['rho_by_scenario']]}  vs baseline {[round(x, 3) for x in r.get('minus_baseline_ci', [])]}")
    (out_dir / "eval.json").write_text(json.dumps(out, indent=1))
    (out_dir / "eval.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-20:]))
    return out
