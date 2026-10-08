# Review of ideas_local.html (5 Oct 2026)

Checked against the 7 REPORT.md files, LIT.md and the JSON files. The judge section was ignored.
Almost every table number matches its source. The problems are below.

## 1. Numbers
- Idea 4 "What we ran": "17 features". dryrun/fmea REPORT says 15. Fix: "15 features".
- Cross-references are off by one. Idea 7 is section 8 and metrics are section 9. Chart caption "see section 7" -> section 8. Glossary "Procedure level ... (section 8)" -> section 9. Section 1 "procedure-level metric in section 8" -> section 9.
- Idea 2 "split-half 0.89 on SOPBench and 0.87 on τ²". Raw split-half is 0.79 and 0.77. 0.89 and 0.87 are after Spearman-Brown. Fix: say so, or give 0.79 and 0.77.
- Idea 7 gain interval is [0.216, 0.386] in the summary and the text but [0.222, 0.383] in the judge table for e = 0 (200 draws against 500). Fix: add one line saying why.
- Section 9: "0.35 to 0.43 against 0.24 with new procedures". The three named methods give 0.351, 0.382, 0.416. Fix: "0.35 to 0.42".
- Idea 3 table baseline sop-dom shows 0.12, source 0.125 (0.13 if rounded half up). Minor.
- No source in the listed files: "noise ceiling about 0.89", "about 2.6 GB of trajectories", "HANDBOOK.md ... about ±0.26", pooled 0.124 (kNN) and 0.255 (fine-tuned encoder; 0.255 is in rerun/tables.md), "$29, 150 cases" (plan doc), "κ 0.62" (rubric/REPORT.md). Add a source line or drop.

## 2. Significance claims
- "[0.005, 0.190]" for Idea 1 on SOPBench new domain exists only in dryrun/out/run/eval.txt (dry_cat_adele, one scenario). dryrun/REPORT.md says the only intervals above 0 are procedure-level. The lower end is 0.005, in one of 4 scenarios, for one of about 6 dry variants, among about 40 variants in the night. The sop-proc interval is [-0.015, 0.214] and the pooled one is [-0.054, 0.089].
  Sentences to change: "On SOPBench with a new domain this is significant" (Idea 1 Result), "significant with a new domain" (summary table), "significant on SOPBench with a new domain" (section 10).
  Fix: "interval of the difference [0.005, 0.190] in this one split, pooled interval includes 0, not corrected for the many variants".
- Idea 1 Verdict "A real but SOPBench-only gain of about +0.09". The report says "low confidence for any small gain". Fix: "a possible gain".
- Idea 1 "above the procedure-level ceiling of section 1". True for new procedures (0.23 against 0.175) but for new domain 0.21 is below 0.231. Fix: say "above it with new procedures".
- Section 9: "the baseline is clearly not the best documentation method". The rubric interval is [-0.008, 0.370] and the others include 0 in new procedures. Fix: "is below the policy-reading methods on this metric, but most intervals include 0".
- Idea 7 "A large, significant gain" is right for the simulation (lower end 0.216). See 4 for what it omits.

## 3. Citations (LIT.md)
Not in LIT.md (not necessarily wrong): Farquhar et al. 2024 Nature 630 (cited in dryrun/REPORT.md only), Lord 1980, van der Linden and Glas 2010, Wolpert/Vapnik volume numbers.
In LIT.md and matching: Thurstone, Pollitt, Qin, Liusie, Ballon, Vacareanu, Kolesnikova, Wang, Acquaye, IEC 60812, Design Science doi, Vapnik and Vashist, Lopez-Paz, Wolpert, van der Laan, Polo.
- Vacareanu is listed in anchors/REPORT.md as COLM 2024. LIT.md says venue unconfirmed. The page says preprint, which is safe.
- Idea 1: "The literature check found no work that uses such a pre-execution check". LIT.md adds that the keyword search was weak and novelty risk is low to medium. Fix: add that.
- "Agent psychometrics", ADeLe and the protocol (Ge et al. 2026, Zhou et al. Nature 2026, Krsteski and Meyer 2026) are used without a citation. Add them.

## 4. Overclaiming and hidden caveats
- Section 1 and the section 9 "have little room left" / section 10 "cannot go much above 0.2": the 0.231 cap is for features of the procedure only. The page itself shows oracles at 0.52 to 0.61 and Idea 1 at 0.23. Fix: "procedure-level features cannot", and delete "little room left".
- Idea 2 "On τ² they work without any training". The report says "suggestive only", the τ² zero-shot interval is [0.004, 0.459], and telecom is noise. Fix: "show a weak signal on τ² airline and retail".
- Idea 2 Verdict "because the hidden account data decide it" is an explanation, not a test. Fix: "probably because".
- Idea 3 control "0.105 against -0.066": 300 calls, SOPBench new domain only, no interval computed (report says so). Add that.
- Idea 5 summary row and Verdict: LUPI 0.516 is not compared with the plain ridge on the same columns (0.360), paired interval not computed (report). The summary row omits this. Add "one metric among many".
- Idea 7: the page omits that this is a within-benchmark simulation with 15 (τ²) or 28 (SOPBench) agents and that the pilot's ability comes from the fit on training tasks (pilot/REPORT.md section 8). Also omitted: at k = 5 the τ² target loses agents, and the doc and no-doc adaptive rows have different targets. The claim "documentation prior adds nothing measurable" has a half-width of about ±0.08.
- Idea 7 "Adaptive is the best rule at every k": no paired interval between rules was computed, report calls it "suggestive". Fix: add.
- Idea 7 "one run of a cheap agent ... 0.45": 0.453 uses the adaptive pilot, which is often Claude 3.7 Sonnet or gpt-4.1. The cheap-model figure is 0.327. Fix in the intro of section 10 and in plain words of Idea 7.
- Judge: the page says "simulated" in section 8 but section 10 says "about 0.30 with a judge that errs one time in five" with no "simulated". The flips are symmetric and independent of the case. Asymmetric errors were not simulated (report). Add.
- "a second pilot recovers most of the loss" is wrong. At e = 0.2, k = 2 gives 0.382, perfect k = 1 gives 0.453, k = 1 at e = 0.2 gives 0.305. It recovers about half. Fix: "about half".
- Omitted: pilot/REPORT.md section 10. At procedure level, with the clean truth, piloting does not improve the ranking of procedures. Matters because the page makes procedure-level ρ a primary metric. Add one sentence.
- Multiple comparisons: only stated in the last sentence of the page. Add one sentence to the intro, since the sections use "significant".
- 3 seeds in one run: stated. Pilot target excluding pilots: stated.

## 5. Terms used before explained
In the intro and the table, before the glossary: ρ, ADeLe demand levels, Agent psychometrics, SOPBench, τ²-bench, "improvement rule", "criterion 4 of the rule" (never defined), log-loss skill, ridge, β and "IRT scale", pooled ρ, Bradley-Terry, Rasch, AUC, "temperature 1", semantic entropy, oracle (defined in section 1 after first use in the table), "your router idea", "the plan", HANDBOOK.md, ThinkingBox-Bench, JEV-as-a-Judge, E8. Intro says "Sections 1 to 9" and there are 10. Fix: one-sentence definitions in the glossary and a pointer in the intro.

## 6. Style (visible text, outside style and script)
- Semicolons: 4, all in citation lists (Idea 2 Why it could work x2, Idea 5, Idea 7). Fix: replace with " and " or separate sentences.
- Em dashes: 0. En dashes: 0.
- "not X but Y": none found. Contrast with "not": section 9 Proposal "to evaluate the router, not to pick a difficulty model". Rephrase: "use them only to evaluate the router".
- Sales or strong words: "Very reliable scores", "large gain", "A large, significant gain", "clearly", "A real but ... gain", "much harder". Replace with the numbers.
- Stacked hedges: none found.
