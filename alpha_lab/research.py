"""Orchestrate exploration, proposal, evaluation, and bounded revision."""
import ast
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from . import VERSION
from .backtest import evaluate, split
from .context import Brief, FirmPolicy
from .literature import retrieve
from .planner import Candidate, PlannerUnavailable
from .signal import InvalidSignal, compile_signal


def diagnose(metrics, rebalance_every):
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
    enough_data = all(
        result["rank_ic_days"] >= result["minimum_ic_days"] and result["mean_coverage"] >= 0.8
        for result in metrics.values()
    )
    if (rebalance_every == 1 and later and enough_data
            and later["gross_bps_per_day"] > 0 >= later["net_bps_per_day"]):
        return {"decision": "revise", "reasons": reasons, "suggestion": "test_rebalance_cadence"}
    return {"decision": "reject", "reasons": reasons, "suggestion": "try_different_mechanism"}


def run_research(panel, registry, objective, notes, planner, provider, budget, cost_bps,
                 *, seed=None, reveal_holdout=False, brief=None, policy=None, market=None):
    research, validation, holdout = split(panel)
    brief = brief or Brief(objective)
    policy = policy or FirmPolicy()
    if cost_bps < policy.minimum_cost_bps:
        raise ValueError(f"firm policy requires at least {policy.minimum_cost_bps:g} bps per unit traded")
    allowed = tuple(name for name in policy.allowed_features if name in brief.allowed_features)
    if not allowed:
        raise ValueError("the brief and firm policy have no permitted features in common")
    as_of = panel.dates[research.start]
    literature = retrieve(notes, brief.objective, as_of) if brief.discovery == "literature_first" else []
    context = {
        "objective": brief.objective,
        "discovery": brief.discovery,
        "features": panel.explore(validation.end, allowed),
        "literature": literature,
        "firm_policy": asdict(policy),
        "brief": asdict(brief),
        "market_context": market,
        "context_as_of": as_of.isoformat(),
        "data_schema": panel.schema,
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
                {key: item[key] for key in ("expression", "decision", "reasons", "rebalance_every")}
                for item in registry.recent(scope)
            ],
            "last_attempt": last,
        }
        canonical = None
        stop_after_attempt = False
        cadence = 1
        if last and last.get("suggestion") == "test_rebalance_cadence":
            control = last["expression"] + " | rebalance=2"
            if not registry.has_attempt(scope, control):
                cadence = 2
        try:
            if cadence == 2:
                candidate = Candidate(**last["candidate"])
            else:
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
            supporting = retrieve(notes, f"{candidate.expression} {candidate.thesis} {brief.objective}", as_of)
            event = {
                "candidate": asdict(candidate), "expression": candidate.expression, "rebalance_every": cadence,
                "evidence_review": supporting,
            }
            try:
                tree = compile_signal(candidate.expression, allowed=allowed, allow_models=policy.allow_models)
            except InvalidSignal as exc:
                event.update(decision="invalid_signal", reasons=[str(exc)])
            else:
                expression = ast.unparse(tree)
                event["expression"] = expression
                plan_key = expression if cadence == 1 else expression + " | rebalance=2"
                thesis = candidate.thesis.lower()
                if expression == "rank(volume_shock)" and ("news" in thesis or "information flow" in thesis):
                    event["pretest_flags"] = ["Unsigned activity alone cannot identify the direction of news flow."]
                if cadence == 2:
                    event["experiment_plan"] = {
                        "control_step": last["step"], "change": "rebalance every second session",
                        "fixed": "same signal, data windows, target and costs",
                        "success": "validation turnover falls and the candidate passes the same development screen",
                    }
                if seed is None and registry.has_attempt(scope, plan_key):
                    event.update(decision="duplicate", reasons=["Expression already evaluated for this data and cost policy"])
                else:
                    canonical = plan_key
                    metrics = {
                        window.name: evaluate(panel, tree, window, cost_bps, rebalance_every=cadence)
                        for window in (research, validation)
                    }
                    event.update(diagnose(metrics, cadence), metrics=metrics)
                    if cadence == 2:
                        before, after = last["metrics"]["validation"], metrics["validation"]
                        event["control_comparison"] = {
                            "turnover_change": round(after["mean_turnover"] - before["mean_turnover"], 6),
                            "net_bps_change": round(after["net_bps_per_day"] - before["net_bps_per_day"], 6),
                            "rank_ic_unchanged": after["rank_ic"] == before["rank_ic"],
                        }
                        if event["decision"] == "shortlist" and after["mean_turnover"] >= before["mean_turnover"]:
                            event.update(decision="reject", reasons=["validation: cadence did not reduce turnover"])
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
                metrics = evaluate(
                    panel, compile_signal(selected["expression"]), holdout, cost_bps,
                    rebalance_every=selected["rebalance_every"],
                )
                screen = diagnose({"holdout": metrics}, selected["rebalance_every"])
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
    print(f"Discovery: {report['discovery']} | Retrieved before proposal: {notes or 'none'}")
    if report["market_context"]:
        snapshot = report["market_context"]
        print(f"Market context ({snapshot['available_on']}, {snapshot['source']}): {snapshot['summary']}")
    print("Decision-day feature coverage:")
    for feature, details in report["features"].items():
        print(f"  {feature}: {details['decision_day_coverage']:.1%}")
    for event in report["attempts"]:
        print(f"\n{event['step']}. {event.get('expression', '(no expression)')} → {event['decision']}")
        if "candidate" in event:
            print(f"   Thesis: {event['candidate']['thesis']}")
            print(f"   Falsifier: {event['candidate']['falsifier']}")
        if event.get("evidence_review"):
            print("   Related literature: " + ", ".join(item["id"] for item in event["evidence_review"]))
        for flag in event.get("pretest_flags", []):
            print(f"   Evidence check: {flag}")
        if "experiment_plan" in event:
            print("   Control: same scores, rebalance every second session.")
        for window, metrics in event.get("metrics", {}).items():
            print(f"   {window}: IC {metrics['rank_ic']}, gross {metrics['gross_bps_per_day']:+.2f} bps/day, "
                  f"net {metrics['net_bps_per_day']:+.2f}, stress {metrics['stress_net_bps_per_day']:+.2f}, "
                  f"turnover {metrics['mean_turnover']:.2f}×")
        for reason in event["reasons"]:
            print(f"   - {reason}")
        if event.get("suggestion") == "test_rebalance_cadence":
            print("   Next experiment: keep the scores fixed and change only the rebalance cadence.")
    print(f"\nResult: {report['conclusion']} | Holdout: {report['holdout']['status']}")
    print(f"Stop: {report['stop_reason']} | Report: {path}")
