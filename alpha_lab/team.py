"""Four collaborating research roles. Tools own data access, arithmetic and gates."""
import ast
import hashlib
import json
import math
from uuid import uuid4

from . import VERSION
from .agent_runtime import AgentFailure, AgentLoop, Finish, Tool
from .backtest import evaluate, split
from .catalog import derive
from .context import market_snapshot
from .literature import retrieve
from .research import diagnose
from .signal import InvalidSignal, compile_signal


def screen_metrics(metrics, capabilities, cadence):
    fields = {"rank_ic_days", "minimum_ic_days", "mean_coverage", "rank_ic",
              "net_bps_per_day", "stress_net_bps_per_day"}
    if capabilities.get("net_costs") is not True or not capabilities.get("cost_model"):
        return ["net cost methodology is unverified"]
    for result in metrics.values():
        if not fields <= result.keys() or any(
            not (key == "rank_ic" and result[key] is None)
            and (type(result[key]) not in (int, float) or not math.isfinite(result[key]))
            for key in fields
        ):
            return ["backtest result lacks required coverage or cost evidence"]
    return diagnose(metrics, cadence)["reasons"]


class LocalBacktest:
    """Replace this interface with the firm's documented backtest adapter."""

    def capabilities(self):
        return {"provider": "synthetic local simulator", "net_costs": True,
                "cost_model": "flat bps per unit traded", "cadence": [1, 2, 3, 4, 5],
                "execution": "decision close; enter next open; daily open-to-open marking"}

    def run(self, panel, tree, window, cost_bps, cadence):
        return evaluate(panel, tree, window, cost_bps, rebalance_every=cadence)


class DataDesk:
    def __init__(self, catalog, policy, allowed):
        self.catalog, self.policy, self.allowed = catalog, policy, allowed
        self.panels, self.recipes = {}, {}

    def search(self, query):
        return self.catalog.search(query, self.policy.permits_dataset)

    def permitted_features(self, source_id):
        return tuple(name for name in self.panels[source_id].registry if (
            (self.allowed is None or name in self.allowed) or name in self.recipes.get(source_id, {})
        ))

    def inspect(self, source_id):
        if not self.policy.permits_dataset(source_id):
            raise ValueError("dataset is outside the firm policy")
        if source_id not in self.panels:
            self.panels[source_id] = self.catalog.load(source_id)
        panel = self.panels[source_id]
        research = split(panel)[0]
        return {"source_id": source_id, "format": panel.schema,
                "features": panel.explore(research.end, self.permitted_features(source_id))}

    def build(self, source_id, output, source, operation, window):
        if source_id not in self.panels or source not in self.permitted_features(source_id):
            raise ValueError("inspect an approved dataset and use an approved input before deriving features")
        recipes = self.recipes.setdefault(source_id, {})
        if len(recipes) >= 3:
            raise ValueError("feature preparation budget reached")
        self.panels[source_id] = derive(self.panels[source_id], output, source, operation, window)
        recipes[output] = {"input": source, "operation": operation, "window": window}
        return {"feature": output, **self.inspect(source_id)}

    def select(self, source_id, rationale):
        if source_id not in self.panels or not self.permitted_features(source_id) or not rationale.strip():
            raise ValueError("inspect a usable source and explain the selection")
        return Finish({**self.inspect(source_id), "rationale": rationale,
                       "recipes": self.recipes.get(source_id, {}).copy()})

    def tools(self):
        return {
            "search_catalog": Tool("Discover permitted sources by research question.", {"query": str}, self.search),
            "inspect_source": Tool("Read definitions, release timing, and development feature coverage.", {"source_id": str}, self.inspect),
            "derive_feature": Tool("Create a causal change, relative_to_mean, or carry_forward recipe. Lookback 1–20 sessions. Carry forward only observations already released at each decision close.",
                                   {"source_id": str, "output": str, "source": str, "operation": str, "window": int}, self.build),
            "submit_data": Tool("Hand off an inspected data source and explain why it fits.", {"source_id": str, "rationale": str}, self.select),
        }


def _stop(reason):
    if not reason.strip():
        raise ValueError("explain why the research should stop")
    return Finish({"stop": reason})


def run_team(*, brief, requested_features, policy, catalog, literature, market_records, registry,
             model, provider, rounds, cost_bps, holdout=False, seed=None, backend=None):
    backend = backend or LocalBacktest()
    caps = backend.capabilities()
    if cost_bps < policy.minimum_cost_bps:
        raise ValueError(f"firm policy requires at least {policy.minimum_cost_bps:g} bps per unit traded")
    allowed = policy.allowed_features
    if requested_features is not None:
        allowed = tuple(name for name in requested_features if allowed is None or name in allowed)
    if allowed == ():
        raise ValueError("the brief and firm policy permit no common features")

    base = {
        "objective": brief.objective, "discovery": brief.discovery,
        "policy": policy.document, "unverified_policy": policy.unresolved,
        "backtest": caps, "cost_bps": cost_bps,
    }
    run_id = uuid4().hex
    spec = {"protocol": VERSION, "provider": provider, "attempt_budget": rounds, **base}
    registry.start(run_id, spec)
    team = AgentLoop(model, limit=min(100, rounds * 26))
    attempts, data_record = [], None
    final = {"status": "not_requested"}
    outcome, stop = "no_candidate", "experiment_budget"
    selected = None
    panel = None

    def experiment_scope():
        return hashlib.sha256(json.dumps(
            [VERSION, panel.sha256, cost_bps, policy.document, caps], sort_keys=True
        ).encode()).hexdigest()

    def read_memory():
        if panel is None:
            raise ValueError("select data before reading matching experiments")
        return [{key: item[key] for key in ("expression", "cadence", "decision", "reasons")}
                for item in registry.recent(experiment_scope())]

    def select_data(question):
        nonlocal data_record, panel
        desk = DataDesk(catalog, policy, allowed)
        data_record = team.run(
            "Data & Feature Agent",
            "Find a source using the catalog, inspect availability and quality. Derive features only when justified. "
            "Hand off a documented source or stop. Features and releases are evidence; never assume missing metadata.",
            {**base, "question": question},
            {**desk.tools(), "stop": Tool("Report an unmet data prerequisite.", {"reason": str}, _stop)},
        )
        if "stop" in data_record:
            panel = None
            return data_record
        panel = desk.panels[data_record["source_id"]]
        data_record = {**data_record, "data_sha256": panel.sha256,
                       "market_context": market_snapshot(market_records, panel.dates[0])}
        return data_record

    try:
        for step in range(1, rounds + 1):
            last = attempts[-1] if attempts else None
            evidence = {}

            def search_notes(query):
                if panel is None:
                    raise ValueError("identify a research dataset first so evidence can be filtered by date")
                passages = retrieve(literature, query, panel.dates[0])
                evidence.update((passage["id"], passage) for passage in passages)
                return passages

            def submit_hypothesis(thesis, falsifier):
                if panel is None or not thesis.strip() or not falsifier.strip():
                    raise ValueError("supply verified data, a thesis, and a concrete falsifier")
                return Finish({"thesis": thesis, "falsifier": falsifier, "evidence": list(evidence.values())})

            control = last is not None and last["review"]["action"] == "execution_control"
            if control:
                hypothesis = last["hypothesis"]
                plan = {"expression": last["expression"], "cadence": last["review"]["cadence"],
                        "rationale": last["review"]["reason"], "parent_step": last["step"]}
            elif seed is not None:
                select_data("Find a permitted daily panel to evaluate the supplied candidate.")
                if panel is None:
                    stop = "data_unavailable"
                    break
                hypothesis = {"thesis": seed.thesis, "falsifier": seed.falsifier,
                              "evidence": [p for p in literature if p["id"] in seed.evidence_ids
                                           and p["available_on"] < panel.dates[0].isoformat()]}
                plan = {"expression": seed.expression, "cadence": 1, "rationale": "supplied candidate"}
            else:
                hypothesis = team.run(
                    "Lead Researcher",
                    "Own the objective. Delegate data discovery, choose whether to search literature, and formulate "
                    "one economic thesis and an observation that would falsify it. Use prior results to select "
                    "a genuinely useful next experiment. Stop if the necessary inputs or ideas are unavailable.",
                    {**base, "last_attempt": last, "data": data_record,
                     "remaining_experiments": rounds - step + 1},
                    {
                        "delegate_data": Tool("Assign data discovery and feature preparation.", {"question": str}, select_data),
                        "search_literature": Tool("Retrieve dated literature passages with citations.", {"query": str}, search_notes),
                        "read_memory": Tool("Read prior evaluated expressions on the selected source and policy.", {}, read_memory),
                        "submit_hypothesis": Tool("Hand off one falsifiable thesis for implementation.",
                                                  {"thesis": str, "falsifier": str}, submit_hypothesis),
                        "stop": Tool("Finish the research with a reason.", {"reason": str}, _stop),
                    },
                )
                if "stop" in hypothesis:
                    stop = hypothesis["stop"]
                    break

                def check_signal(expression, cadence):
                    if cadence not in caps.get("cadence", (1,)):
                        raise ValueError("requested execution cadence is not supported")
                    tree = compile_signal(expression, allowed=data_record["features"], registry=panel.registry,
                                          allow_models=policy.allow_models)
                    return {"expression": ast.unparse(tree), "cadence": cadence}

                def submit_signal(expression, cadence, rationale):
                    if not rationale.strip():
                        raise ValueError("explain how the expression tests the thesis")
                    return Finish({**check_signal(expression, cadence), "rationale": rationale})

                plan = team.run(
                    "Signal Implementation Agent",
                    "Translate the thesis into rank(term). Use documented feature names, negation, addition, "
                    "subtraction, lag(feature,1–10), mean(feature,2–10) or rank(ridge(feature,feature,15–40)) "
                    "when policy permits. Check the expression before handing it off. Stop if the thesis "
                    "cannot be represented with the available inputs.",
                    {**base, "hypothesis": hypothesis, "data": data_record, "last_attempt": last,
                     "memory": read_memory()},
                    {
                        "check_signal": Tool("Validate feature permissions, expression and execution settings.",
                                             {"expression": str, "cadence": int}, check_signal),
                        "submit_signal": Tool("Hand off a checked signal and execution plan.",
                                              {"expression": str, "cadence": int, "rationale": str}, submit_signal),
                        "stop": Tool("Report an implementation blocker.", {"reason": str}, _stop),
                    },
                )
                if "stop" in plan:
                    stop = plan["stop"]
                    break

            event = {"step": step, "hypothesis": hypothesis, "data": data_record, **plan}
            scope = experiment_scope()
            registry_key = None
            try:
                tree = compile_signal(plan["expression"], allowed=data_record["features"],
                                      registry=panel.registry, allow_models=policy.allow_models)
            except InvalidSignal as exc:
                event.update(decision="invalid_signal", reasons=[str(exc)],
                             review={"action": "reject", "reason": str(exc), "cadence": 1})
            else:
                event["expression"] = ast.unparse(tree)
                key = f"{event['expression']} | cadence={plan['cadence']}"
                prior = registry.recent(scope)
                if seed is None and registry.has_attempt(scope, key):
                    reason = "This expression and execution rule have already been evaluated on this data."
                    event.update(decision="duplicate", reasons=[reason],
                                 review={"action": "revise", "reason": reason, "cadence": 1})
                else:
                    registry_key = key
                    research, validation, final_window = split(panel)
                    metrics = {window.name: backend.run(panel, tree, window, cost_bps, plan["cadence"])
                               for window in (research, validation)}
                    event["metrics"] = metrics
                    reasons = screen_metrics(metrics, caps, plan["cadence"])
                    reasons += [f"unverified firm policy: {item}" for item in policy.unresolved]

                    def inspect_window(window):
                        if window not in metrics:
                            raise ValueError("only development results are available to this agent")
                        return {"window": window, **metrics[window]}

                    def inspect_costs(window):
                        result = inspect_window(window)
                        return {"model": caps.get("cost_model"), "net_costs": caps.get("net_costs"), **{
                            key: result.get(key) for key in ("gross_bps_per_day", "net_bps_per_day",
                                                             "stress_net_bps_per_day", "mean_turnover")
                        }}

                    def submit_review(action, reason, cadence):
                        if action not in ("shortlist", "reject", "revise", "execution_control", "stop") or not reason.strip():
                            raise ValueError("submit a supported decision with its evidence")
                        if action == "execution_control" and (cadence == plan["cadence"]
                                or cadence not in caps.get("cadence", ())
                                or registry.has_attempt(scope, f"{event['expression']} | cadence={cadence}")):
                            raise ValueError("control must choose a supported, untried execution cadence")
                        return Finish({"action": action, "reason": reason, "cadence": cadence})

                    try:
                        review = team.run(
                            "Validation & Critic Agent",
                            "Use only development evidence. Identify the strongest limitation and decide whether to "
                            "reject, revise the hypothesis, test the same score with another cadence, shortlist, or stop. "
                            "An execution control changes cadence only. Never treat unknown cost or policy as a pass.",
                            {**base, "experiment": event, "gates": reasons,
                             "prior_experiments": [{k: item[k] for k in ("expression", "cadence", "decision")}
                                                   for item in prior if "cadence" in item],
                             "remaining_experiments": rounds - step},
                            {
                                "inspect_window": Tool("Inspect metrics and dates for research or validation.", {"window": str}, inspect_window),
                                "inspect_costs": Tool("Inspect cost convention, turnover and gross/net results.", {"window": str}, inspect_costs),
                                "submit_review": Tool("Choose shortlist, reject, revise, execution_control or stop.",
                                                      {"action": str, "reason": str, "cadence": int}, submit_review),
                            },
                        )
                    except AgentFailure as exc:
                        outcome = "agent_stopped"
                        review = {"action": "stop", "reason": str(exc), "cadence": plan["cadence"]}
                    decision = "blocked_shortlist" if review["action"] == "shortlist" and reasons else review["action"]
                    event.update(review=review, decision=decision, reasons=reasons)
            registry.record(run_id, scope, step, registry_key, event)
            attempts.append(event)
            if event["decision"] == "shortlist":
                selected = event
                outcome, stop = "candidate_for_independent_review", "shortlisted"
                if holdout:
                    def verify_frozen():
                        result = backend.run(panel, tree, final_window, cost_bps, plan["cadence"])
                        passed = not screen_metrics({"holdout": result}, caps, plan["cadence"])
                        return {"status": "evaluated", "passed_screen": passed, "metrics": result}

                    final = registry.open_holdout(panel.sha256, run_id, verify_frozen)
                    if final.get("passed_screen") is False:
                        outcome = "selected_candidate_failed_holdout"
                break
            if event["review"]["action"] == "stop":
                stop = event["review"]["reason"]
                break
    except AgentFailure as exc:
        stop = str(exc)
        outcome = "agent_stopped"
    if holdout and selected is None:
        final = {"status": "no_eligible_candidate"}
    report = {**spec, "run_id": run_id, "conclusion": outcome, "stop_reason": stop,
              "holdout": final, "selected": selected, "attempts": attempts,
              "agent_trace": team.trace, "actions_used": min(100, rounds * 26) - team.remaining}
    registry.finish(run_id, report)
    return report


def show_team_report(report, path):
    print(f"Objective: {report['objective']}")
    print(f"Provider: {report['provider']} | Backtest: {report['backtest']['provider']}")
    for item in report["agent_trace"]:
        marker = " [error]" if isinstance(item["result"], dict) and "error" in item["result"] else ""
        print(f"  {item['role']} → {item['tool']}{marker}")
    for event in report["attempts"]:
        print(f"\n{event['step']}. {event['expression']} → {event['decision']}")
        print(f"   Thesis: {event['hypothesis']['thesis']}")
        print(f"   Execution cadence: every {event['cadence']} session(s)")
        for window, metrics in event.get("metrics", {}).items():
            print(f"   {window}: IC {metrics.get('rank_ic')}, gross {metrics.get('gross_bps_per_day')}, "
                  f"net {metrics.get('net_bps_per_day')}, turnover {metrics.get('mean_turnover')}")
        print(f"   Critic: {event['review']['reason']}")
        for reason in event["reasons"]:
            print(f"   - {reason}")
    print(f"\nResult: {report['conclusion']} | Holdout: {report['holdout']['status']}")
    print(f"Stop: {report['stop_reason']} | Report: {path}")
