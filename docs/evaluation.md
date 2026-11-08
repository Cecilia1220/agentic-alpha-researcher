# External system evaluation and interview walkthrough

## Prototype demonstration

Start with a research objective. The Lead delegates data preparation, the Data & Feature Agent selects and profiles a source, the Signal Implementation Agent submits a validated expression, and the Critic selects a follow-up. In the synthetic reversal example, the Critic keeps the signal fixed and trades less often. Turnover falls, but net performance remains negative, so the candidate is rejected. The separate seeded case demonstrates rejection of a future label before backtesting.

The trace records the input and output of each tool call. The local run uses a reproducible offline policy. Model-driven research uses the same tool and handoff contracts with an approved API.

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
