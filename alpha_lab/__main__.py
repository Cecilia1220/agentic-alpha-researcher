"""Local CLI. Run from the repository root."""
import argparse
import json
import math
import os
from pathlib import Path

from .context import Brief, FirmPolicy, market_snapshot, read_json
from .data import load_panel
from .literature import validate_corpus
from .planner import Candidate, DEFAULT_MODEL, OfflinePlanner, OpenAIPlanner
from .registry import Registry
from .research import print_report, run_research, write_report


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Run a bounded, local alpha research loop.")
    task = parser.add_mutually_exclusive_group()
    task.add_argument("--objective", help="Free-text research objective")
    task.add_argument("--brief", type=Path, help="JSON objective and discovery route")
    parser.add_argument("--literature", type=Path, default=ROOT / "data" / "literature.json")
    parser.add_argument("--firm-policy", type=Path, help="JSON asset scope, cost floor and feature permissions")
    parser.add_argument("--market-context", type=Path, help="JSON dated background snapshots")
    parser.add_argument("--provider", choices=("offline", "openai"), default="offline")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic_panel.csv")
    parser.add_argument("--registry", type=Path, default=Path("runs/research.sqlite3"))
    parser.add_argument("--reports", type=Path, default=Path("runs/reports"))
    parser.add_argument("--rounds", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--cost-bps", type=float, default=12.0)
    parser.add_argument("--seed", type=Path, help="Validate and, if valid, backtest exactly one supplied candidate")
    parser.add_argument("--holdout", action="store_true", help="Evaluate one selected candidate once per dataset in this registry")
    args = parser.parse_args()
    if not math.isfinite(args.cost_bps) or not 0 <= args.cost_bps <= 100:
        parser.error("cost must be between 0 and 100 bps")
    if args.seed is not None and (args.provider == "openai" or args.holdout):
        parser.error("seeded experiments are local and cannot open the holdout")
    try:
        brief = Brief.parse(
            read_json(args.brief) if args.brief else
            {"objective": args.objective if args.objective is not None else "Explore daily cross-sectional relative returns."}
        )
        policy = FirmPolicy.parse(read_json(args.firm_policy)) if args.firm_policy else FirmPolicy()
        notes = validate_corpus(read_json(args.literature))
        panel = load_panel(args.data)
        market = market_snapshot(read_json(args.market_context), panel.dates[0]) if args.market_context else None
        seed = None
        if args.seed is not None:
            value = json.loads(args.seed.read_text(encoding="utf-8"))
            seed = Candidate.parse(value, {note["id"] for note in notes})
        provider = f"openai/{args.model}" if args.provider == "openai" else "offline"
        planner = OpenAIPlanner(args.model) if args.provider == "openai" else OfflinePlanner()
        report = run_research(
            panel, Registry(args.registry), brief.objective, notes, planner,
            "seeded" if seed else provider, 1 if seed else args.rounds, args.cost_bps,
            seed=seed, reveal_holdout=args.holdout, brief=brief, policy=policy, market=market,
        )
        print_report(report, write_report(report, args.reports))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 1 if report["stop_reason"] == "provider_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
