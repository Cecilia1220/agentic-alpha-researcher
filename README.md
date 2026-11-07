# Agentic Alpha Researcher

A local research prototype for Trexquant SuperDay Topic B. It turns a research brief into an auditable series of daily equity signal experiments. Each candidate includes an economic thesis, an executable scoring rule, a falsifier, and an evaluation trace. A run can finish without a candidate if the evidence falls short.

## Quick start

Python 3.10 or newer. The offline planner, data loader, feature recipes, and backtester use the Python standard library. From the repository root:

```sh
demo_dir="$(mktemp -d)"

# Retrieve ideas first, then evaluate a signal and its execution control.
python3 -m alpha_lab --objective "Study short-horizon reversal and liquidity pressure" \
  --holdout --registry "$demo_dir/main.sqlite3" --reports "$demo_dir/reports"

# Start from data and a directed brief, then retrieve relevant literature.
python3 -m alpha_lab --brief examples/research_brief.json \
  --firm-policy examples/firm_policy.json --market-context examples/market_context.json \
  --rounds 1 --registry "$demo_dir/brief.sqlite3" --reports "$demo_dir/reports"

# Fit a supervised score using only labels available by each decision.
python3 -m alpha_lab --objective "Test a ridge model" \
  --rounds 1 --registry "$demo_dir/model.sqlite3" --reports "$demo_dir/reports"

# Exercise the hard look-ahead gate.
python3 -m alpha_lab --seed examples/leaky_signal.json \
  --registry "$demo_dir/main.sqlite3" --reports "$demo_dir/reports"

python3 -m unittest discover -s tests -v
```

The first run proposes daily reversal. In validation its gross result is **+2.79 bps/day**, and its net result is **−19.10 bps/day**. The critic schedules a paired experiment. It keeps the signal, data, label and costs fixed and rebalances every second session. Validation turnover falls from **1.82× to 0.91×** with the same daily Rank IC, while gross P&L falls to **−9.54 bps/day** and net P&L falls to **−20.50 bps/day**. The control fails. The next hypothesis also fails the screen. The run records the failures and leaves the holdout window unevaluated. All figures come from the bundled synthetic dataset.

The default registry is `runs/research.sqlite3`. Later runs use it to avoid repeating experiments with the same expression, execution rule, data and cost assumptions. Use a fresh registry to reproduce the demonstration. JSON reports default to `runs/reports/`.

## Inputs and roles

- **Research brief.** A free-text objective can be broad or directed. A JSON brief can also choose `literature_first` or `data_first` and narrow the allowed feature set.
- **Firm policy.** The tool supports daily equities. The policy can restrict features, require a minimum cost assumption, or disallow fitted models. The compiler enforces the intersection of brief and policy features. The JSON files demonstrate local enforcement. A hosted deployment should load firm policy from configuration controlled by the firm.
- **Market context.** An optional file supplies dated summaries and source names. The planner receives the most recent snapshot available strictly before the first research session. This is background for forming hypotheses. Tradable market context would need to enter through the point-in-time feature store at each decision date. The bundled snapshots are synthetic.
- **Literature retrieval.** Supply a JSON corpus with `--literature`. Each record is a short passage or research note with an ID, title, source, availability date, keywords and text. The retriever returns up to two matching passages per query, filtered by the research start date. Literature-first retrieves from the objective before proposing. Data-first begins with the objective and feature profile. Both routes retrieve again using the thesis and expression and log the excerpts for evidence review. The offline planner chooses from explicit ideas, while the optional OpenAI planner can propose new expressions. Retrieval uses lexical matches, so related wording and citation scope still require review.
- **Research loop.** The planner proposes, the validator checks the expression, and the backtester computes results. The critic applies fixed acceptance rules and can schedule a controlled change to execution cadence. The evidence review flags the specific case where a volume-only score is used to infer the direction of news. SQLite records the run and attempted plans, including blocked proposals, diagnoses and controls. The attempt limit is one to five per run.

## Data, features and scoring

The bundled panel has 96 weekday sessions and eight fictitious assets. The simulation plants weak reversal early and removes it later. Regenerate it with `python3 data/generate.py`.

- **Raw CSV.** One row per `date,symbol`, with `open,close,volume,value_score,value_available`. The panel is rectangular and needs at least 40 sessions and six symbols. The loader derives one- and five-session close-to-close returns and current volume divided by the preceding five-session mean, minus one. The precomputed-feature CSV format (`ret_1d,ret_5d,volume_shock` in place of `close,volume`) is also accepted. Precomputed features must already obey their stated as-of contract.
- **Availability.** Prices, returns and volume features used for scoring are assumed available by the decision close. The report value has its own release date. The signal accessor checks the date on every read. Missing or unreleased values are excluded. Source timestamps, corporate actions and universe history must be supplied by a production data store.
- **Formula score.** Expressions have the form `rank(term)`, where terms are registered features, negation, sums or differences, positive lags and trailing means. For example, `rank(-ret_1d)` ranks recent losers higher. `rank(-mean(ret_1d, 3))` is a different signal with a three-session smoothing rule. No arbitrary code runs.
- **Fitted score.** `rank(ridge(ret_1d, volume_shock, 20))` learns the weights of two within-day ranked predictors using a fixed ridge penalty. At decision close `t`, the most recent permitted training label is for the decision on `t-2`, whose return ends at the open of `t`. The model trains on up to 20 completed decision dates and requires ten valid cross-sections before producing a score. It predicts the next tradable return relative to the contemporaneous cross-section, and its daily prediction ranks are the alpha scores. The sample model misses the coverage screen during warm-up and also fails later net performance.

## Backtest and decisions

Decide after the close on `t`, enter at the open of `t+1`, and measure the open-to-open return ending at `t+2`. The simulator buys and shorts the top and bottom thirds, rounded down, with target notional of 0.5 per side. It splits cutoff ties and records opening, rebalancing and liquidation turnover against fixed reference notional. The cadence control carries marked holdings between scheduled rebalances, subject to liquidating on unusable scores. Daily Rank IC always compares the original daily scores with one-day labels.

The chronological 60/20/20 split purges the last two decision dates in each block. Research and validation both inform iteration. To shortlist, the fixed demo screen requires at least 80% score coverage and valid IC days, Rank IC of at least 0.02, and positive net P&L at base and doubled costs in both development blocks. The default cost is 12 bps per unit traded. These are demonstration thresholds. The prose falsifier is recorded separately from the numerical screen.

With `--holdout`, the runner evaluates at most one frozen candidate per dataset per local registry and withholds its result from the planner. A frozen rolling model may use earlier labels that have matured by a later holdout decision, as specified before evaluation. The CSV includes all three windows. Independent verification requires separately controlled data and access.

The sample backtest assumes a fixed universe and omits market impact, borrow availability, realistic auction fills, factor neutrality and capacity. The ranking and P&L are synthetic. A strategy that passes the local screen would still need independent evidence.

## Optional OpenAI planner

Set `OPENAI_API_KEY` securely in the shell environment. Then run:

```sh
python3 -m alpha_lab --provider openai --model gpt-4o-mini --rounds 2
```

The planner uses the Responses API with Structured Outputs and `store: false`. It can send the objective, brief, policy, feature summaries, retrieved excerpts, the dated market summary and recent development feedback to OpenAI. Supply only material you are authorized to transmit. Keep credentials out of Git and reports. Requests are capped at one per proposal with a 600-output-token limit and a 25-second socket timeout. The automated test uses a simulated response. The offline commands provide the reproducible demo.

[OpenAI Structured Outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)

See [External system evaluation and interview walkthrough](docs/evaluation.md).
