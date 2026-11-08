# Agentic Alpha Researcher

A multi-agent research prototype for Trexquant SuperDay Topic B. Given a research objective, the team discovers eligible data, proposes and implements a daily cross-sectional signal, requests a backtest, and decides what to investigate next. The output includes an economic thesis, executable signal, experiment configuration, evidence, and a decision. A run can conclude without an eligible candidate.

## Quick start

Python 3.10 or newer. Run from the repository root. The local demonstration uses the Python standard library and synthetic data.

```sh
demo_dir="$(mktemp -d)"
python3 -m alpha_lab --objective "Study short-horizon reversal and liquidity pressure" \
  --firm-policy examples/firm_policy.json --market-context examples/market_context.json \
  --holdout --registry "$demo_dir/research.sqlite3" --reports "$demo_dir/reports"
python3 -m alpha_lab --seed examples/leaky_signal.json \
  --registry "$demo_dir/research.sqlite3" --reports "$demo_dir/reports"
python3 -m unittest discover -s tests -v
```

The trace shows which role called each tool. The offline provider uses a small reproducible decision policy to exercise the same roles, tools, handoffs, and controls as the model provider. In the first run, the Critic requests an execution control that keeps the signal fixed and trades less often. Turnover falls in validation, but net performance remains negative and the candidate is rejected. The seeded run submits a future-return label and is blocked before backtesting.

To let the agents choose tools and construct their own proposals, set `OPENAI_API_KEY` securely in the shell environment and run:

```sh
python3 -m alpha_lab --provider openai --objective "Study short-horizon reversal and liquidity pressure" \
  --firm-policy examples/firm_policy.json --registry "$demo_dir/model.sqlite3" --reports "$demo_dir/reports"
```

Use `--model` or `OPENAI_MODEL` to select a compatible Responses API model. The same model can serve different roles. Each role receives its own instructions and tool list. The harness caps actions per role and run and uses a 25-second socket timeout per model request. The application sends the research context, policy, retrieved excerpts, feature summaries, and development feedback to OpenAI with `store: false`. Use approved data, provider credentials, and access controls. The automated tests simulate model responses. [OpenAI function calling documentation](https://developers.openai.com/api/docs/guides/function-calling)

## Research team

1. **Lead Researcher** owns the objective and remaining experiment budget. It can delegate data discovery, search dated literature, consult matching experiment history, hand off a falsifiable economic thesis, or stop.
2. **Data & Feature Agent** searches the authorized catalog, inspects data definitions and development coverage, and chooses a source. It can ask tools to construct causal features, such as a trailing return, relative activity, or a bounded carry-forward of released values. It hands off the selected source, available features, recipe, and rationale.
3. **Signal Implementation Agent** turns the thesis into a supported expression and execution plan. It can check and repair the expression before submitting it. The interpreter enforces registered inputs, point-in-time access and permitted operations. It also supports a small rolling ridge model if policy permits it.
4. **Validation & Critic Agent** receives development results and applicable evidence gates. It can inspect a window or cost convention and recommend a new hypothesis, a same-score execution control, rejection, shortlisting, or stopping. Independent checks gate any shortlist. Holdout results are kept out of agent context.

The harness validates requests and hands results between roles, records attempts and tool evidence, and enforces budget and stop conditions. The signal interpreter and backtest engine compute reproducible results. SQLite holds experiment memory. The JSON report contains the thesis, signal and experiment configuration, data fingerprint, feature lineage, policy, metrics, decision, and tool trace.

## Inputs and data access

- **Research request.** Use `--objective` for a question or `--brief examples/research_brief.json` for an objective, optional discovery preference, and optional feature restriction. The CLI supplies a broad default objective. A chat interface or scheduled research queue could submit the same brief.
- **Company policy.** Use `--firm-policy` with JSON or a text document. The original input is retained. Supported controls include permitted datasets and features, a cost floor, and permission to fit models. Unknown controls and unverified text are surfaced and block shortlisting. Additional controls require a corresponding enforcement tool. The installed catalog defines which sources the local tools can read.
- **Catalog.** Use `--catalog` with a list of source descriptions, locations, key mappings, and feature metadata. The built-in adapters support CSV, JSON records, and read-only SQLite tables. A new source or documented feature can be registered without editing the agent code. Availability is declared as session close or a mapped release-date column. Sources are restricted to the catalog directory. The local simulator requires one selected panel with complete execution prices and a fixed universe. Cross-source joins can be supplied as a curated view. `--data` also accepts the original raw or precomputed daily-equity CSV formats.
- **Literature and market context.** `--literature` points to attributed, dated notes. The Lead can retrieve relevant passages before a hypothesis or use them to check an idea after inspecting data. The optional `--market-context` supplies dated summaries for hypothesis formation. The harness selects an eligible snapshot before the start of the research window. It does not turn that background into a tradable feature.

See [the bundled catalog](data/catalog.json) and [policy example](examples/firm_policy.json) for concrete input contracts.

## Backtest contract and production assumptions

The demo generates 96 synthetic weekday sessions for eight fictitious assets. Regenerate the panel with `python3 data/generate.py`. A score formed at close `t` can trade at open `t+1`. The next-open return ending at `t+2` is the one-day evaluation label. The local evaluator reports Rank IC, coverage, turnover, gross P&L, net P&L, and a doubled-cost scenario. The cost model is a configurable flat number of basis points per unit traded. The research, validation, and holdout windows are chronological. With `--holdout`, at most one frozen candidate per dataset is evaluated within a local registry. A production holdout should be access-controlled separately.

The production design assumes a **read-only company database or feature service** with documented point-in-time data, an **approved literature corpus**, a **company backtest service**, and a **company-approved LLM API with a finite budget**. The company provides the authorized universe, policy and execution conventions. The local CSV and simulator let the prototype run independently of those services.

The backtest interface has a capability description and an evaluation operation. A firm adapter would submit the signal or daily scores with the universe, date window, execution convention, and cost settings. It would map the returned metrics and methodology into a reviewable result. If the firm service supplies net metrics with documented costs, the Critic uses those values. If the service provides only gross performance, a documented cost model and adequate trading data are needed to assess net performance. The harness blocks a shortlist when net cost evidence is unavailable. The repository includes the local simulator and the adapter boundary. Connecting a firm service requires its actual API contract and permissions.

The fixture demonstrates the research process with synthetic observations. Production use requires market-impact and borrow models where relevant, risk and capacity checks, independently controlled verification data, and firm review.

See [External system evaluation](docs/evaluation.md) for the presentation plan covering human researcher comparisons, correctness, research quality, efficiency, and system improvement.
