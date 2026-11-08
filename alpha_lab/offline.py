"""Reproducible local policy exercising the same agent tools without API calls."""
from .literature import terms


class OfflineTeam:
    def choose(self, role, instruction, context, tools, history):
        seen = {item["tool"]: item["result"] for item in history}

        def call(name, **args):
            return name, args, [], ""

        objective = context["objective"].lower()
        momentum = bool(terms(objective) & {"momentum", "trend", "continuation"})
        if role == "Lead Researcher":
            if context["last_attempt"]:
                return call("stop", reason="The bounded offline research policy has no further distinct thesis.")
            if "delegate_data" not in seen:
                return call("delegate_data", question=context["objective"])
            data = seen["delegate_data"]
            if "stop" in data or "error" in data:
                return call("stop", reason=data.get("stop") or data["error"])
            if context["discovery"] == "literature_first" and "search_literature" not in seen:
                return call("search_literature", query=context["objective"])
            if "recent_price_change" not in data["features"]:
                return call("submit_hypothesis",
                            thesis="The documented feature may contain information about relative future returns.",
                            falsifier="The feature fails to rank subsequent returns reliably after accounting for costs.")
            return call(
                "submit_hypothesis",
                thesis=("Gradual information diffusion may cause recent winners to continue outperforming."
                        if momentum else "Temporary price pressure may reverse, so recent losers could subsequently outperform."),
                falsifier="Relative predictive power and after-cost performance do not persist across development windows.",
            )
        if role == "Data & Feature Agent":
            if "search_catalog" not in seen:
                return call("search_catalog", query=context["question"])
            candidates = seen["search_catalog"]
            inspected = {item["arguments"]["source_id"] for item in history if item["tool"] == "inspect_source"}
            result = seen.get("inspect_source")
            if result is None or "error" in result or not result["features"]:
                for item in candidates:
                    if item["id"] not in inspected:
                        return call("inspect_source", source_id=item["id"])
                return call("stop", reason="No authorized source passed the data checks.")
            source_id = result["source_id"]
            if "derive_feature" not in seen:
                for name, feature in result["features"].items():
                    if "closing price" in feature["description"].lower():
                        return call("derive_feature", source_id=source_id, output="recent_price_change",
                                    source=name, operation="change", window=5 if momentum else 1)
            return call("submit_data", source_id=source_id,
                        rationale="The selected source has documented availability and usable development coverage.")
        if role == "Signal Implementation Agent":
            features = context["data"]["features"]
            feature = "recent_price_change" if "recent_price_change" in features else max(
                features, key=lambda name: features[name]["decision_day_coverage"]
            )
            direction = "-" if feature == "recent_price_change" and not momentum else ""
            args = {"expression": f"rank({direction}{feature})", "cadence": 1}
            if "check_signal" not in seen:
                return call("check_signal", **args)
            if "error" in seen["check_signal"]:
                return call("stop", reason=seen["check_signal"]["error"])
            return call("submit_signal", **args, rationale="Rank eligible stocks in the direction predicted by the thesis.")
        if "inspect_window" not in seen:
            return call("inspect_window", window="validation")
        observed = seen["inspect_window"]
        experiment = context["experiment"]
        cadence = experiment["cadence"]
        if not context["gates"]:
            return call("submit_review", action="shortlist", reason="The development screens passed.", cadence=cadence)
        if ("inspect_costs" not in seen and observed.get("gross_bps_per_day") is not None
                and observed.get("net_bps_per_day") is not None
                and observed["gross_bps_per_day"] > 0 >= observed["net_bps_per_day"]):
            return call("inspect_costs", window="validation")
        prior = {item["cadence"] for item in context["prior_experiments"]
                 if item["expression"] == experiment["expression"]}
        if (context["remaining_experiments"] and cadence == 1
                and 2 in context["backtest"]["cadence"] and 2 not in prior
                and observed.get("gross_bps_per_day") is not None and observed.get("net_bps_per_day") is not None
                and observed["gross_bps_per_day"] > 0 >= observed["net_bps_per_day"]):
            return call("submit_review", action="execution_control", cadence=2,
                        reason="Gross performance turned negative after costs. Retest the same scores with fewer rebalances.")
        return call("submit_review", action="reject", cadence=cadence,
                    reason="The evidence did not clear the development requirements.")
