import copy
import csv
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from alpha_lab.backtest import _weights, evaluate, rank_ic, split
from alpha_lab.context import Brief, FirmPolicy, market_snapshot
from alpha_lab.data import load_panel
from alpha_lab.literature import retrieve
from alpha_lab.planner import Candidate, OfflinePlanner, OpenAIPlanner, PlannerUnavailable
from alpha_lab.registry import Registry
from alpha_lab.research import run_research
from alpha_lab.signal import InvalidSignal, compile_signal, ranks, score

ROOT = Path(__file__).resolve().parents[1]


class PrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = load_panel(ROOT / "data/synthetic_panel.csv")
        cls.notes = json.loads((ROOT / "data/literature.json").read_text())

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.registry = Registry(Path(folder.name) / "research.sqlite3")

    def run_demo(self, **kwargs):
        options = {
            "objective": "Explore daily cross-sectional relative returns.",
            "notes": self.notes, "planner": OfflinePlanner(), "provider": "offline",
            "budget": 3, "cost_bps": 12.0,
        }
        options.update(kwargs)
        return run_research(self.panel, self.registry, **options)

    def test_autonomous_failure_revision_and_memory(self):
        report = self.run_demo(objective="Study short-horizon reversal and liquidity pressure", reveal_holdout=True)
        original, revised, unrelated = report["attempts"]
        self.assertEqual([item["decision"] for item in report["attempts"]], ["revise", "reject", "reject"])
        self.assertEqual(revised["follows_step"], 1)
        self.assertEqual(unrelated["expression"], "rank(volume_shock)")
        before, after = (item["metrics"]["validation"] for item in (original, revised))
        self.assertGreater(before["gross_bps_per_day"], 0)
        self.assertLess(after["mean_turnover"], before["mean_turnover"])
        self.assertEqual(before["rank_ic"], after["rank_ic"])
        self.assertEqual(original["expression"], revised["expression"])
        self.assertEqual(revised["rebalance_every"], 2)
        self.assertGreater(before["net_bps_per_day"], after["net_bps_per_day"])
        self.assertIn("Unsigned", unrelated["pretest_flags"][0])
        self.assertLess(after["net_bps_per_day"], 0)
        self.assertIsNone(report["selected"])
        self.assertEqual(report["holdout"]["status"], "no_eligible_candidate")
        rerun = self.run_demo()
        self.assertEqual([item["expression"] for item in rerun["attempts"]], ["rank(ret_5d)"])
        self.assertEqual(rerun["stop_reason"], "no_novel_offline_hypothesis")

    def test_objective_changes_initial_proposal(self):
        report = self.run_demo(objective="Test a short-horizon momentum hypothesis", budget=1)
        self.assertEqual(report["attempts"][0]["expression"], "rank(ret_5d)")

    def test_two_retrieval_routes_and_date_boundaries(self):
        objective = "Study short-horizon momentum"
        literature = self.run_demo(objective=objective, budget=1)
        self.assertEqual(literature["literature"][0]["id"], "momentum")
        future = dict(self.notes[2], id="later", available_on="2025-01-01")
        self.assertNotIn("later", [p["id"] for p in retrieve(self.notes + [future], objective, self.panel.dates[0])])
        data_first = self.run_demo(
            brief=Brief("Test abnormal volume", "data_first"), budget=1,
        )
        self.assertEqual(data_first["literature"], [])
        self.assertEqual(data_first["attempts"][0]["evidence_review"][0]["id"], "volume")
        snapshots = [
            {"available_on": "2023-12-01", "summary": "Older", "source": "test"},
            {"available_on": "2024-01-02", "summary": "Later", "source": "test"},
        ]
        self.assertEqual(market_snapshot(snapshots, date(2024, 1, 2))["summary"], "Older")

    def test_policy_and_brief_constraints_reach_the_compiler(self):
        candidate = Candidate("Momentum", "rank(ret_5d)", "No IC", [])
        policy = FirmPolicy.parse({"allowed_features": ["ret_1d"], "minimum_cost_bps": 12, "allow_models": False})
        with self.assertRaisesRegex(ValueError, "requires at least"):
            self.run_demo(policy=policy, cost_bps=0)
        with patch("alpha_lab.research.evaluate") as backtest:
            event = self.run_demo(policy=policy, seed=candidate, budget=1)["attempts"][0]
            backtest.assert_not_called()
        self.assertIn("firm policy disallows", event["reasons"][0])
        with self.assertRaisesRegex(ValueError, "no permitted features"):
            self.run_demo(policy=policy, brief=Brief("Momentum", allowed_features=("ret_5d",)))
        with self.assertRaisesRegex(InvalidSignal, "disallows fitted models"):
            compile_signal("rank(ridge(ret_1d, ret_5d, 20))", allow_models=False)

    def test_raw_feature_recipes_and_model_use_only_matured_labels(self):
        session, symbol = 30, self.panel.symbols[0]
        today = self.panel.rows[self.panel.dates[session], symbol]
        previous = self.panel.rows[self.panel.dates[session - 1], symbol]
        volume_base = sum(self.panel.rows[day, symbol]["volume"] for day in self.panel.dates[session - 5:session]) / 5
        self.assertAlmostEqual(today["ret_1d"], today["close"] / previous["close"] - 1)
        self.assertAlmostEqual(today["volume_shock"], today["volume"] / volume_base - 1)
        expr = compile_signal("rank(ridge(ret_1d, volume_shock, 20))")
        self.assertEqual(score(self.panel, expr, 3), {})
        historical_return = self.panel.forward_return
        with patch.object(type(self.panel), "forward_return", autospec=True,
                          side_effect=lambda _, day, name: historical_return(day, name)) as labels:
            expected = score(self.panel, expr, session)
        self.assertEqual(max(call.args[1] for call in labels.call_args_list), session - 2)
        self.assertEqual(len(expected), len(self.panel.symbols))
        revised = replace(self.panel, rows=copy.deepcopy(self.panel.rows))
        for (day, name), row in revised.rows.items():
            if day > self.panel.dates[session]:
                row["open"] *= 1 + 0.1 * self.panel.symbols.index(name)
                row["volume_shock"] = -999
        self.assertEqual(score(revised, expr, session), expected)
        report = self.run_demo(objective="Test ridge model", budget=1)
        self.assertIn("metrics", report["attempts"][0])
        self.assertIn("ridge", report["attempts"][0]["expression"])

    def test_precomputed_csv_remains_supported(self):
        fields = ["date", "symbol", "open", "ret_1d", "ret_5d", "volume_shock", "value_score", "value_available"]
        path = self.registry.path.with_suffix(".csv")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for (day, symbol), row in self.panel.rows.items():
                writer.writerow({"date": day, "symbol": symbol, **row})
        legacy = load_panel(path)
        self.assertEqual(legacy.schema, "features")
        expr = compile_signal("rank(-ret_1d)")
        self.assertEqual(score(legacy, expr, 20), score(self.panel, expr, 20))

    def test_seeded_label_is_rejected_before_backtest(self):
        seed = Candidate.parse(json.loads((ROOT / "examples/leaky_signal.json").read_text()), set())
        with patch("alpha_lab.research.evaluate") as backtest:
            report = self.run_demo(seed=seed, budget=1)
            backtest.assert_not_called()
        event = report["attempts"][0]
        self.assertEqual(event["decision"], "invalid_signal")
        self.assertIn("leak future", event["reasons"][0])
        self.assertNotIn("metrics", event)

    def test_grammar_rejects_untrusted_or_forward_expressions(self):
        for expression in (
            "rank(open)", "rank(lag(ret_1d, -1))", "rank(mean(ret_1d, 100))",
            "rank(lag(ret_1d, True))", "rank(x[0])", "rank(unknown)",
            "rank(__import__('os').getcwd())", "rank(ret_1d, ret_5d)", "ret_1d",
        ):
            with self.subTest(expression=expression), self.assertRaises(InvalidSignal):
                compile_signal(expression)

    def test_pit_release_and_insensitivity_to_future_mutation(self):
        self.assertEqual(score(self.panel, compile_signal("rank(value_score)"), 10), {})
        expr = compile_signal("rank(lag(value_score, 2))")
        expected = score(self.panel, expr, 10)
        self.assertEqual(len(expected), len(self.panel.symbols))
        revised = replace(self.panel, rows=copy.deepcopy(self.panel.rows))
        for (session, _), row in revised.rows.items():
            if row["value_available"] > self.panel.dates[10]:
                row["value_score"] = -999
            if session > self.panel.dates[10]:
                row["open"] = 1000
        self.assertEqual(score(revised, expr, 10), expected)

    def test_ties_and_rank_ic(self):
        tied = ranks({s: 0 for s in self.panel.symbols})
        self.assertEqual(set(tied.values()), {0.5})
        self.assertEqual(set(_weights(tied, self.panel.symbols).values()), {0.0})
        returns = dict(zip(self.panel.symbols, range(len(self.panel.symbols))))
        self.assertIsNone(rank_ic(tied, returns))
        self.assertAlmostEqual(rank_ic(ranks(returns), returns), 1)

    def test_alignment_and_trading_costs_by_hand(self):
        window = replace(split(self.panel)[0], start=1, end=4)
        expr = compile_signal("rank(-ret_1d)")
        scores = score(self.panel, expr, 1)
        weights = _weights(scores, self.panel.symbols)
        prices = self.panel.rows
        entry, exit_ = self.panel.dates[2:4]
        pnl = sum(w * (prices[exit_, s]["open"] / prices[entry, s]["open"] - 1) for s, w in weights.items())
        liquidation = sum(abs(w) * prices[exit_, s]["open"] / prices[entry, s]["open"] for s, w in weights.items())
        result = evaluate(self.panel, expr, window, 12)
        self.assertEqual(result["days"], 1)
        self.assertAlmostEqual(result["mean_turnover"], 1 + liquidation, places=5)
        self.assertAlmostEqual(result["net_bps_per_day"], pnl * 10_000 - 12 * (1 + liquidation), places=5)

    def test_holdout_is_opened_once_and_not_seen_by_planner(self):
        class CapturingPlanner(OfflinePlanner):
            def propose(self, context):
                self.context = context
                return super().propose(context)

        planner = CapturingPlanner()
        report = self.run_demo(planner=planner, cost_bps=0, budget=1, reveal_holdout=True)
        self.assertIsNotNone(report["selected"])
        self.assertEqual(report["holdout"]["status"], "evaluated")
        self.assertFalse(report["holdout"]["passed_screen"])
        self.assertEqual(report["conclusion"], "selected_candidate_failed_holdout")
        self.assertNotIn("holdout", json.dumps(planner.context))
        with patch("alpha_lab.research.evaluate", side_effect=AssertionError("must not reopen")) as backtest:
            status = self.registry.open_holdout(self.panel.sha256, report["run_id"], backtest)
            backtest.assert_not_called()
        self.assertEqual(status["status"], "already_opened")

    def test_openai_wire_contract_and_parsing_without_network(self):
        context = {"literature": self.notes}
        proposal = {"thesis": "Thesis", "expression": "rank(ret_5d)", "falsifier": "Test",
                    "evidence_ids": ["momentum"]}
        payload = {"status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": json.dumps(proposal)}]},
        ]}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only"}):
            planner = OpenAIPlanner("test-model")
        with patch("alpha_lab.planner.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as call:
            result = planner.propose(context)
        req = call.call_args.args[0]
        sent = json.loads(req.data)
        self.assertEqual(result.expression, proposal["expression"])
        self.assertEqual(req.full_url, "https://api.openai.com/v1/responses")
        self.assertFalse(sent["store"])
        self.assertEqual(sent["max_output_tokens"], 600)
        self.assertTrue(sent["text"]["format"]["strict"])
        with patch("alpha_lab.planner.request.urlopen", side_effect=HTTPError(req.full_url, 429, "", {}, None)):
            with self.assertRaisesRegex(PlannerUnavailable, "HTTP 429"):
                planner.propose(context)

    def test_provider_failure_is_recorded_and_stops(self):
        class FailingPlanner:
            def propose(self, context):
                raise PlannerUnavailable("OpenAI returned HTTP 429")

        report = self.run_demo(planner=FailingPlanner())
        self.assertEqual(report["stop_reason"], "provider_error")
        self.assertEqual(len(report["attempts"]), 1)
        self.assertEqual(report["attempts"][0]["decision"], "provider_error")


if __name__ == "__main__":
    unittest.main()
