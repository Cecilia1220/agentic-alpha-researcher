"""Orchestrate exploration, proposal, evaluation, and bounded revision."""
import ast
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from . import VERSION
from .backtest import evaluate, split
from .planner import PlannerUnavailable, retrieve_notes
from .signal import InvalidSignal, compile_signal


def diagnose(metrics, expression):
    """Fixed screening policy for the synthetic demonstration."""
    reasons = []
    for window, result in metrics.items():
        if result["rank_ic_days"] < result["minimum_ic_days"] or result["mean_coverage"] < 0.8:
            reasons.append(f"{window}: insufficient ranked observations")
        if result["rank_ic"] is None or result["rank_ic"] < 0.02:
            reasons.append(f"{window}: rank IC below +0.02")
        if result["net_bps_per_day"] <= 0:
            reasons.append(f"{window}: net performance at or below zero")
        if result["stress_net_bps_per_day"] <= 0:
            reasons.append(f"{window}: fails 2x cost stress")
    if not reasons:
        return {"decision": "shortlist", "reasons": [], "suggestion": None}
    later = metrics.get("validation")
    if (expression == "rank(-ret_1d)" and later
            and later["gross_bps_per_day"] > 0 >= later["net_bps_per_day"]):
        return {"decision": "revise", "reasons": reasons, "suggestion": "smooth_reversal"}
    return {"decision": "reject", "reasons": reasons, "suggestion": "try_different_mechanism"}


def run_research(panel, registry, objective, notes, planner, provider, budget, cost_bps,
                 *, seed=None, reveal_holdout=False):
    research, validation, holdout = split(panel)
    context = {
        "objective": objective,
        "features": panel.explore(validation.end),
        "literature": retrieve_notes(notes, objective),
        "cost_bps": cost_bps,
    }
    run_id = uuid4().hex
    run_spec = {"protocol": VERSION, "data_sha256": panel.sha256, "provider": provider,
                "attempt_budget": budget, "holdout_requested": reveal_holdout, **context}
    registry.start(run_id, run_spec)
    scope = f"{VERSION}:{panel.sha256}:{float(cost_bps).hex()}"
    attempts = []
    shortlisted = []
    stop = "budget_exhausted"
    for step in range(1, budget + 1):
        last = attempts[-1] if attempts else None
        provider_context = {
            **context, "attempt": step, "remaining_attempts": budget - step,
            "memory": [
                {key: item[key] for key in ("expression", "decision", "reasons")}
                for item in registry.recent(scope)
            ],
            "last_attempt": last,
        }
        canonical = None
        stop_after_attempt = False
        try:
            candidate = seed if seed is not None else planner.propose(provider_context)
        except PlannerUnavailable as exc:
            event = {"decision": "provider_error", "reasons": [str(exc)]}
            stop = "provider_error"
            stop_after_attempt = True
        except ValueError as exc:
            event = {"decision": "invalid_candidate", "reasons": [str(exc)]}
        else:
            if candidate is None:
                stop = "no_novel_offline_hypothesis"
                break
            event = {"candidate": asdict(candidate), "expression": candidate.expression}
            try:
                tree = compile_signal(candidate.expression)
            except InvalidSignal as exc:
                event.update(decision="invalid_signal", reasons=[str(exc)])
            else:
                expression = ast.unparse(tree)
                event["expression"] = expression
                if seed is None and registry.has_attempt(scope, expression):
                    event.update(decision="duplicate", reasons=["Expression already evaluated for this data and cost policy"])
                else:
                    canonical = expression
                    metrics = {w.name: evaluate(panel, tree, w, cost_bps) for w in (research, validation)}
                    event.update(diagnose(metrics, expression), metrics=metrics)
                    if event["decision"] == "shortlist":
                        shortlisted.append(event)
        event["step"] = step
        if last and last.get("decision") == "revise":
            event["follows_step"] = last["step"]
        registry.record(run_id, scope, step, canonical, event)
        attempts.append(event)
        if stop_after_attempt:
            break

    # Selection uses development metrics only. Holdout never returns to the planner.
    selected = max(shortlisted, key=lambda event: min(
        item["stress_net_bps_per_day"] for item in event["metrics"].values()
    ), default=None)
    final = {"status": "not_requested"}
    if reveal_holdout:
        if selected is None:
            final = {"status": "no_eligible_candidate"}
        else:
            def evaluate_selected():
                metrics = evaluate(panel, compile_signal(selected["expression"]), holdout, cost_bps)
                screen = diagnose({"holdout": metrics}, selected["expression"])
                return {"status": "evaluated", "metrics": metrics,
                        "passed_screen": screen["decision"] == "shortlist", "reasons": screen["reasons"]}

            final = registry.open_holdout(panel.sha256, run_id, evaluate_selected)
    outcome = {
        "stop_reason": stop, "selected": selected, "holdout": final, "attempts": attempts,
        "conclusion": "candidate_for_independent_review" if selected else "no_candidate_met_development_screen",
    }
    if final.get("passed_screen") is False:
        outcome["conclusion"] = "selected_candidate_failed_holdout"
    registry.finish(run_id, outcome)
    return {"run_id": run_id, **run_spec, **outcome}


def write_report(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{report['run_id']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def print_report(report, path: Path):
    print(f"Objective: {report['objective']}")
    print(f"Planner: {report['provider']} | Trading cost: {report['cost_bps']:g} bps per unit traded")
    notes = ", ".join(note["id"] for note in report["literature"])
    print(f"Read {notes}. Decision-day feature coverage:")
    for feature, details in report["features"].items():
        print(f"  {feature}: {details['decision_day_coverage']:.1%}")
    for event in report["attempts"]:
        print(f"\n{event['step']}. {event.get('expression', '(no expression)')} → {event['decision']}")
        if "candidate" in event:
            print(f"   Thesis: {event['candidate']['thesis']}")
            print(f"   Falsifier: {event['candidate']['falsifier']}")
        for window, metrics in event.get("metrics", {}).items():
            print(f"   {window}: IC {metrics['rank_ic']}, gross {metrics['gross_bps_per_day']:+.2f} bps/day, "
                  f"net {metrics['net_bps_per_day']:+.2f}, stress {metrics['stress_net_bps_per_day']:+.2f}, "
                  f"turnover {metrics['mean_turnover']:.2f}×")
        for reason in event["reasons"]:
            print(f"   - {reason}")
        if event.get("suggestion") == "smooth_reversal":
            print("   Next experiment: smooth the return input and measure turnover and after-cost performance.")
    print(f"\nResult: {report['conclusion']} | Holdout: {report['holdout']['status']}")
    print(f"Stop: {report['stop_reason']} | Report: {path}")
