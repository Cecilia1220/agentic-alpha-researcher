"""Local CLI. Run from the repository root."""
import argparse
import math
import os
from pathlib import Path

from .agent_runtime import OpenAIToolModel
from .catalog import Catalog
from .context import Brief, read_json
from .literature import validate_corpus
from .offline import OfflineTeam
from .planner import Candidate, DEFAULT_MODEL
from .policy import Policy
from .registry import Registry
from .research import write_report
from .team import run_team, show_team_report


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
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--data", type=Path, help="Use one CSV with the built-in daily equity schema")
    sources.add_argument("--catalog", type=Path, help="Use a local catalog of documented data sources")
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
        brief_fields = (read_json(args.brief) if args.brief else
                        {"objective": args.objective if args.objective is not None else
                         "Explore daily cross-sectional relative returns."})
        brief = Brief.parse(brief_fields)
        policy = Policy.load(args.firm_policy)
        notes = validate_corpus(read_json(args.literature))
        if args.data:
            path = args.data.resolve()
            catalog = Catalog([{"id": "supplied_csv", "description": "Supplied daily equity panel",
                                "format": "legacy_csv", "path": path.name}], path.parent)
        else:
            catalog = Catalog.from_file(args.catalog or ROOT / "data" / "catalog.json")
        market = read_json(args.market_context) if args.market_context else []
        seed = Candidate.parse(read_json(args.seed), {note["id"] for note in notes}) if args.seed else None
        provider = f"openai/{args.model}" if args.provider == "openai" else "offline"
        model = (OpenAIToolModel(os.environ.get("OPENAI_API_KEY"), args.model)
                 if args.provider == "openai" else OfflineTeam())
        report = run_team(
            brief=brief, requested_features=brief_fields.get("allowed_features"), policy=policy,
            catalog=catalog, literature=notes, market_records=market, registry=Registry(args.registry),
            model=model, provider="seeded" if seed else provider, rounds=1 if seed else args.rounds,
            cost_bps=args.cost_bps, holdout=args.holdout, seed=seed,
        )
        show_team_report(report, write_report(report, args.reports))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 1 if report["conclusion"] == "agent_stopped" else 0


if __name__ == "__main__":
    raise SystemExit(main())
