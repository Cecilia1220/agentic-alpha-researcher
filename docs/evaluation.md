# External system evaluation and interview walkthrough

## Prototype demonstration

Use the fresh-registry commands in the README. The broad objective retrieves three local reading notes and shows 0% contemporaneous coverage for the delayed feature. The first experiment proposes a falsifiable liquidity-pressure hypothesis and evaluates it at a tradeable clock. In validation it generates positive gross P&L and negative after-cost P&L. Smoothing lowers churn, but the revised signal still fails the predeclared screen. The third candidate also fails. Activity is unsigned, so a claim about positive information flow would need directional data or a better economic mechanism. The seeded forward-label case separately shows a hard safety gate.

The run demonstrates a signal revision under a fixed screening policy. System improvement would require a separate comparison. For example, if the planner repeatedly treats positive gross P&L as robust evidence, add an explicit evidence and cost-stress requirement, then compare versions on the same blinded suite.

## Compare with a human quant researcher

Use practicing QRs, with several researchers and tasks where feasible. Match research briefs, authorized data and literature, backtest tools, time box, and required output. Randomize task assignment or counterbalance paired tasks to avoid familiarity effects. Use blind reviewers and record both machine runtime and reviewer or cleanup time. The primary operational comparison is the distribution of **human QR versus agent-assisted workflow** time to an auditable result that clears a quality bar, including a justified rejection. Report incomplete and disqualified runs as such. Measure active researcher time and elapsed wall time separately. Fix the baseline before comparing versions.

## Score four dimensions

- **Correctness.** Hard gates first, because good results are unusable if timing or arithmetic is wrong. Track the severe-error escape rate and false-rejection rate on valid controls. Combine seeded adversarial cases with generated perturbations of valid panels and seeded examples from past failures. Move release timestamps forward, mutate future rows, permute ticker order, and reconcile turnover and P&L with a reference implementation. If a violation escapes, fix the feature lineage, validator, or evaluator and add a minimal regression. Any observed severe leakage escape blocks promotion of a version.
- **Research quality.** Blind reviewers grade economic mechanism, evidence fit, falsifiability, robustness checks, and whether a rejection is well justified. Track agreement and report sample counts. Compare candidates that pass correctness gates with predeclared, independently controlled tests and report coverage across research tasks. If the agent repeatedly extrapolates the wrong horizon from papers, require a source-scope check and compare blinded rubric scores on unseen topics.
- **Efficiency.** Track time to a reviewable report, researcher and reviewer minutes, repeat work, valid outputs per task, model tokens, and tool calls. Use comparable resource and deadline budgets. If duplication dominates, improve memory and use expression equivalence or signal correlation, then check that the system still explores diverse mechanisms.
- **Improvement.** Blindly score whether a revision addresses the diagnosed failure and whether it preserves validity. Compare the full loop against the same version with critique feedback removed under a matched experiment budget to isolate the effect of feedback. Separately compare whole versions against the human QR baseline. If revisions keep chasing noise, require a specific falsifier and one declared change per revision, tighten the search budget, and test on fresh dates and tasks.

## Iterate as system owner

Freeze a development suite and a separately controlled evaluation suite. Build the cases by failure family and sample additional inputs from observed research traffic, including valid counterexamples. Version the prompts, feature contracts, backtester, fixture seeds, budgets, and rubric. Use repeated model runs and paired task-level comparisons to account for run variance. Triage failures by severity and frequency, change one component, rerun regressions, and evaluate the release candidate on blinded tasks. If a holdout informs a redesign, rotate to a fresh one before making a new release claim. A monitoring agent could tag traces and propose failure cases. Independent reviewers and deterministic checks own the scores and release decision.

This is a presentation plan. The prototype's unit tests cover specific safety and backtest contracts. It does not collect human benchmarks or claim measured improvements across system versions.
