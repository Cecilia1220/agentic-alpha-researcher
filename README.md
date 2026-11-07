# Agentic Alpha Researcher

A local prototype for Trexquant SuperDay Topic B. Give it a research objective and a daily equity panel. It reads a small literature corpus, profiles feature availability, proposes a falsifiable hypothesis with an executable signal, evaluates the signal, and uses the diagnosis to decide what to try next. Every attempt is recorded.

## Run the demo

Python 3.10 or newer. Run from the repository root. The offline mode uses Python's standard library and the included synthetic CSV.

```sh
demo_dir="$(mktemp -d)"
python3 -m alpha_lab --holdout --registry "$demo_dir/research.sqlite3" --reports "$demo_dir/reports"
python3 -m alpha_lab --seed examples/leaky_signal.json --registry "$demo_dir/research.sqlite3" --reports "$demo_dir/reports"
python3 -m unittest discover -s tests -v
```

The first command starts from a broad objective and proposes three candidates. The first ranks yesterday's losers higher. Its later validation window shows **+2.79 bps/day gross and −19.10 bps/day after costs**. The critic proposes smoothing over three sessions. Validation turnover falls from **1.82× to 1.21×** and net performance improves to **−9.66 bps/day**. The revised signal still fails the screen. The run closes without a shortlist and leaves the holdout sealed. The separate seeded command rejects the forward-return label before calling the backtester.

To start with another research objective, use a fresh registry.

```sh
python3 -m alpha_lab --objective "Test a short-horizon momentum hypothesis" --rounds 2
```

The default registry at `runs/research.sqlite3` is persistent and later runs skip already evaluated expressions on the same dataset and cost policy. JSON reports in `runs/reports/` include input provenance, hypotheses, implementations, metrics, decisions, and stopping conditions. A fresh registry reproduces the offline research trace and metrics with a new run ID.

## Architecture and data contracts

1. **Explorer.** The input is a research objective (up to 400 characters), a registered daily CSV, a bounded attempt count, and a fixed cost policy. The explorer ranks three curated notes by objective keywords and profiles point-in-time feature coverage over development dates. The planner sees note provenance and eligible feature summaries.
2. **Research planner.** One provider proposes a candidate with `thesis`, `expression`, `falsifier`, and `evidence_ids`. The offline provider picks from four explicit hypotheses or variants using the objective, coverage, prior tests, and the last diagnosis. The optional model provider can propose new expressions within the same grammar.
3. **Implementation and validation.** The signal is a daily cross-sectional rank score. The grammar is `rank(term)`. A term may be a registered feature, its negation, a sum or difference of terms, `lag(feature, n)` for 1–10 sessions, or `mean(feature, n)` for 2–10 sessions. The interpreter uses an allowlisted syntax tree and a release-aware feature accessor. Reserved execution fields, targets, unknown names, forward lags, and arbitrary code are rejected.
4. **Backtester and critic.** The deterministic evaluator returns mean daily Spearman rank IC, IC coverage, score coverage, turnover, and gross, net, and doubled-cost P&L. The critic records which fixed checks failed and gives the next planner call the diagnosis. Shortlisting requires mean IC of at least 0.02, at least 80% mean score coverage and 80% valid IC days, and positive net and doubled-cost performance in **both** development windows. These thresholds and the default 12 bps cost are illustrative demo choices.
5. **Memory and stopping.** SQLite stores each run and attempt, including rejections and reasons. The latest 12 evaluated expressions in the same data/protocol/cost scope go into the next proposal context. Canonical syntax prevents repeat evaluations in serial runs, with a full lookup beyond the context limit. The runner stops on budget, exhaustion of the offline policy, or a provider error. It ranks eligible candidates by the worse development-window doubled-cost result. A finished output includes an auditable candidate with a thesis, falsifier, expression, evidence IDs and metrics, or a report explaining why the run produced no candidate.

### CSV and execution clock

The synthetic file has 96 weekday sessions and eight fictitious assets. It is deterministic and can be regenerated with `python3 data/generate.py`. The simulation deliberately plants weak short-run reversal in the early window and removes it later. The planner receives coverage and experiment results, with no access to the generating process.

- **Key and units.** Each row is one `date,symbol` pair. A rectangular, fixed-universe panel has at least 40 sessions and six assets. `open` is a positive price. `ret_1d` and `ret_5d` are decimal close-to-close returns known by session close. `volume_shock` is a synthetic activity score known by close. `value_score` uses `value_available`, an ISO date assumed available by that date's close. Blank feature values are missing.
- **Point in time.** Lags count observed sessions. Each read checks that its observation date and release date are no later than the decision close. In the sample, the value score is released two sessions later, so `rank(value_score)` has zero decision-day coverage and `rank(lag(value_score, 2))` can score after warm-up. Trailing means require all observations. Unknown releases and upstream timestamp and lineage correctness are the data provider's responsibility.
- **Target and portfolio.** Score at close `t`, enter at open `t+1`, and measure the open-to-open return ending at `t+2`. Rank IC uses within-day midranks on available scores and realized returns. Fewer than six scores or a constant score gives no positions. The strategy buys the top third and shorts the bottom third, rounded down, at 0.5 target notional per side. Cutoff ties split the allocation, and opposing tied positions net out. Turnover is traded notional as a fraction of fixed reference notional, accounting for price drift, first entry, and final liquidation. Costs scale linearly with traded notional.
- **Time splits.** Chronological 60/20/20 research, validation, and holdout by session count. The last two decision dates in each block are purged so both execution endpoints stay inside the block. Lookback features may use eligible earlier observations. Both development windows inform revision and selection. With `--holdout`, at most one frozen candidate may be evaluated once per dataset per registry. A fresh registry can reopen the fixture, so independent live evaluation requires separately controlled data and permissions.

## Optional OpenAI planner

Set `OPENAI_API_KEY` securely in your shell environment. A shell that supports it can prompt without echoing or putting the key in shell history.

```sh
python3 -m alpha_lab --provider openai --model gpt-4o-mini --rounds 2
```

Use `--model` or `OPENAI_MODEL` to choose a Responses API model that supports Structured Outputs. The implementation makes at most one request per round with a 600-output-token cap, at most five rounds, and a 25-second socket timeout. API calls may be billed. The planner transmits the objective, bundled notes, aggregate feature coverage, and compact development experiment feedback. It uses `store: false`. Keep proprietary information out of the objective and experiment context unless you are authorized to send it. Keep the API key out of reports, commands shared with others, and Git. The included test exercises the wire contract with a simulated API response. The offline demo is the reproducible interview path.

[OpenAI Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)

## Scope and limitations

This fixture demonstrates research mechanics. Its rank IC and P&L are synthetic. The cost model omits market impact, auction fill risk, borrow, liquidity constraints, fees by asset, and portfolio risk controls. The panel assumes a fixed universe, clean opening prices, and trustworthy feature lineage. The prototype tests release timing at date granularity. A production feature store would need as-of joins, source timestamps, revision and universe history, and independent auditing. The paper notes seed ideas and document scope mismatches. They provide no support for the performance of these particular signals. The local registry uses syntactic deduplication. Parallel runs may still evaluate the same expression, and economically equivalent expressions can differ syntactically.

See [External system evaluation and interview walkthrough](docs/evaluation.md).
