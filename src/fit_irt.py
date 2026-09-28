"""Full-data 1PL IRT for each benchmark with the Agent psychometrics trainer (swebench_irt/train.py,
unchanged): P(success) = sigmoid(theta_agent - beta_task), py-irt SVI, their default priors and
epochs. Output data/<bench>/irt/1d_1pl/{abilities,items}.csv.

The fitted beta is the rank target of the evaluation and the oracle. Seed 0, except tau-Knowledge
banking, where seed 0 gives NaN parameters and seed 1 is the first seed that converges.

  python src/fit_irt.py [bench ...]
"""

import subprocess
import sys

from common import AP_REPO, BENCHES, DATA

SEEDS = {"sopbench": 0, "tau2": 0, "tauk_banking": 1}


def fit(responses, out_dir, seed, quiet=False):
    subprocess.run([sys.executable, "swebench_irt/train.py", "--dims", "1", "--model", "1pl",
                    "--data_path", str(responses), "--output_dir", str(out_dir), "--seed", str(seed)],
                   cwd=AP_REPO, check=True, capture_output=quiet)


if __name__ == "__main__":
    for b in sys.argv[1:] or BENCHES:
        fit(DATA / b / "responses.jsonl", DATA / b / "irt", SEEDS[b])
        print(b, "done")
