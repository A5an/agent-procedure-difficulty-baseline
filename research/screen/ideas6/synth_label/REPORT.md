# Refuse-cue classifier trained on synthetic labels (agent `synth_label`, round 6, 6 Oct 2026)

Saved by the main session from the agent's final message. Level L0. About 1,080 gemini-3.8-flash calls. Negative result.

AUC against the real refuse label (830 real messages): zero-shot dry run 0.638; tf-idf trained on 1,365 synthetic cases
0.49 to 0.51; gemini few-shot with 8 synthetic examples of the same policy variant 0.553; averaged with dry run 0.600;
few-shot with synthetic databases shown (249-case subset) 0.486. Synthetic message wording does not transfer to real
messages. No extra generation (bar of 0.64 not reached).

Harness (SOPBench, tau2 part copied from c_lad): c_lad + features 0.258 (-0.001 [-0.007, +0.006]), c_lad + sl_p 0.259
(+0.001), features + ADeLe without PLAN and CMP 0.240 (-0.018, significantly worse). No change in rho_given_label or
procedure-level rho.
Verdict: this route does not help, high confidence.
Files: data.csv, build_data.py, step1a.py, fewshot.py, evalauc.py, feats.py, run.py, eval.py, score.py, out/.
