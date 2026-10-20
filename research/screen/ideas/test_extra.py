import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import metrics_extra as X
R = ZR.REPO + "/results"
S = ZR.SCREEN
res = X.report(R, ['adele', 'human_time', 'length', 'lltm_adele', 'knn_router'],
               extra_dirs={'lad2_adele': f'{S}/rubric/out/run', 'grp_proc_adele': f'{S}/rubric/out/run',
                           'finetuned_bge_small': f'{S}/rerun/out_final', 'shares_adele': f'{S}/targets/run'}, write=False)
import json; json.dump(res, open('extra_existing.json', 'w'), indent=1)
print(json.dumps(res['sopbench|new_domain'].get('_oracle_procedure_mean')), json.dumps(res['sopbench|new_domain'].get('_oracle_procedure_x_label')))
