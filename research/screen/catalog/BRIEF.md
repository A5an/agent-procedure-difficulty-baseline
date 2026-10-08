# Brief: full catalogue of every idea we measured (7 Oct 2026)

The user wants ONE page listing every idea the project has tried since the baseline, each with a short and a detailed
explanation in plain Russian, plus its measured numbers: how much better or worse than the baseline (and than c_lad
where available). You extract the records for your share of folders. The main session builds the page.

## Ground truth for comparisons
- Main metric: pooled within-domain case-level Spearman rho over four new-process scenarios (sop-proc = SOPBench new
  procedures, sop-dom = SOPBench new domain, tau2-proc, tau2-dom). Baseline `adele` (Agent psychometrics + 18 ADeLe
  levels) = 0.131 [0.059, 0.206] in the paired scorer (0.131 [0.027, 0.255] in the repo evaluator). By scenario .120 .125 .128 .150.
- c_lad = 0.258 (rounds 3 onward). Procedure level, SOPBench new domain: adele 0.231, c_lad 0.486.
- Take every number VERBATIM from the folder's REPORT.md, PREREG.md, score txt/json or eval files. Never compute a
  new number, never round differently, never guess. If a number is missing, put null and say so in "note".
- Some ideas are not measured on the pooled metric (AUC of a label, procedure-level only, one benchmark only, agent
  success rate, literature scans). Record them with metric_type set accordingly and the actual number in "other_metric".
  Literature scans and councils without measurements: one record each with metric_type "none".

## One record per idea (variants of one idea go into "variants", the headline variant on top)
```json
{
  "id": "r3_plan_tools",
  "round": "r3",            // r0 = baseline repo (28 Sep), s = screen of 4 Oct, r1 = ideas/ night 4-5 Oct,
                            // r2 = ideas2, r3 = ideas3, r4 = ideas4, r5 = ideas5, r6 = ideas6, r7 = ideas7
  "date": "5 окт",
  "folder": "prototype/screen/ideas3/plan_tools",
  "name": "План действий с настоящими инструментами",   // Russian, 2-6 words
  "family": "plan",         // one of: baseline, text, embed, rubric, plan, code, llm_judge, pairwise, emulation,
                            // agents_run, label_read, small_model, psychometric, search, method, literature
  "level": "L1",            // L0, L1, L2, E, reference (agent runs, oracle), none
  "short": "одно предложение по-русски, что это",
  "detail": "3-6 предложений по-русски: что сделали, как считали, что вышло и почему",
  "metric_type": "pooled4", // pooled4, sop_only, tau2_only, proc, label_auc, success_rate, other, none
  "pooled": 0.214,          // pooled4 rho or null
  "scen": [0.0, 0.0, 0.0, 0.0],   // sop-proc, sop-dom, tau2-proc, tau2-dom or null
  "vs_base": 0.083, "vs_base_ci": [0.021, 0.165],  // difference to adele and its paired CI, or nulls
  "vs_clad": null, "vs_clad_ci": null,
  "proc": null,             // SOPBench procedure-level rho if reported
  "other_metric": "строка по-русски с другими числами, если есть",
  "verdict": "лучше значимо", // лучше значимо | лучше незначимо | на уровне | хуже | не на протоколе
  "variants": [{"name": "...", "pooled": 0.0, "vs_base": 0.0, "vs_base_ci": [0, 0], "vs_clad": null, "note": "..."}],
  "source": "file:line or file name the numbers came from",
  "note": "caveats: post hoc, telecom-driven, oracle, uses gold, L2, etc."
}
```
"verdict" rule: "лучше значимо" only if the paired CI vs the baseline lies above 0; "хуже" if the point estimate is
below the baseline; "на уровне" if within about 0.01; otherwise "лучше незначимо". For non-pooled metrics use
"не на протоколе".

## Russian text rules (the user asked for humanized text)
Plain, spoken Russian, like a colleague explaining. Short sentences, verbs, concrete numbers. Explain jargon in passing
(IRT, ADeLe, ρ) the first time it appears in a record. No em dashes or en dashes as connectors (number ranges like
0.5–0.7 are fine). No "не X, а Y" contrasts, no one-line punchlines, no triads for rhythm, no bold, no
канцелярит (является, данный, осуществлять, в рамках, ключевой, важно отметить, стоит отметить, по сути, фактически,
комплексный), no "уникальный", no hype. Say plainly when something failed and why.

## Output
Write the records as a JSON array with Bash (heredoc or python) to prototype/screen/catalog/<your_name>.json and also
return the same JSON in your final message (file writes may be refused). Read only; do not run harnesses, LLMs or GPU.
