import csv
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from alpha_lab.agent_runtime import AgentLoop, Finish, OpenAIToolModel, Tool
from alpha_lab.catalog import Catalog
from alpha_lab.context import Brief
from alpha_lab.offline import OfflineTeam
from alpha_lab.policy import Policy
from alpha_lab.registry import Registry
from alpha_lab.team import LocalBacktest, run_team

ROOT = Path(__file__).resolve().parents[1]


class TeamTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.registry = Registry(self.folder / "experiments.sqlite")
        self.catalog = Catalog.from_file(ROOT / "data/catalog.json")
        self.literature = json.loads((ROOT / "data/literature.json").read_text())

    def run_team(self, **changes):
        args = dict(brief=Brief("Study short-horizon reversal and liquidity pressure"),
                    requested_features=None, policy=Policy.parse({}), catalog=self.catalog,
                    literature=self.literature, market_records=[], registry=self.registry,
                    model=OfflineTeam(), provider="offline", rounds=2, cost_bps=12)
        return run_team(**{**args, **changes})

    def test_roles_take_actions_and_critic_controls_the_next_experiment(self):
        report = self.run_team()
        actions = [(item["role"], item["tool"]) for item in report["agent_trace"]]
        self.assertIn(("Lead Researcher", "delegate_data"), actions)
        self.assertIn(("Data & Feature Agent", "derive_feature"), actions)
        self.assertIn(("Signal Implementation Agent", "check_signal"), actions)
        self.assertIn(("Validation & Critic Agent", "inspect_costs"), actions)
        first, second = report["attempts"]
        self.assertEqual(first["review"]["action"], "execution_control")
        self.assertEqual(second["parent_step"], first["step"])
        self.assertEqual(second["expression"], first["expression"])
        self.assertEqual(second["data"]["data_sha256"], first["data"]["data_sha256"])
        self.assertLess(second["metrics"]["validation"]["mean_turnover"],
                        first["metrics"]["validation"]["mean_turnover"])
        self.assertIsNone(report["selected"])

    def test_new_sources_and_fields_are_discovered_without_changing_code(self):
        with (ROOT / "data/synthetic_panel.csv").open() as stream:
            rows = [{"session": row["date"], "security": row["symbol"], "entry_open": float(row["open"]),
                     "new_column": float(row["close"]), "feed_note": "extra unconsumed field"}
                    for row in csv.DictReader(stream)]
        (self.folder / "other.json").write_text(json.dumps(rows))
        with closing(sqlite3.connect(self.folder / "other.sqlite")) as db, db:
            db.execute("CREATE TABLE daily (session TEXT, security TEXT, entry_open REAL, new_column REAL, feed_note TEXT)")
            db.executemany("INSERT INTO daily VALUES (:session, :security, :entry_open, :new_column, :feed_note)", rows)

        for fmt, path in (("json", "other.json"), ("sqlite", "other.sqlite")):
            with self.subTest(fmt=fmt):
                item = {"id": f"new_{fmt}", "format": fmt, "path": path, "table": "daily",
                        "description": "New price source with different vendor column names",
                        "keys": {"date": "session", "symbol": "security", "open": "entry_open"},
                        "features": {"vendor_price": {"column": "new_column", "availability": "close",
                                                     "description": "Closing price in currency units"}}}
                catalog = Catalog([item], self.folder)
                result = self.run_team(catalog=catalog, rounds=1)["attempts"][0]
                self.assertEqual(result["data"]["source_id"], item["id"])
                self.assertEqual(result["data"]["recipes"]["recent_price_change"]["input"], "vendor_price")
                self.assertIn("validation", result["metrics"])

    def test_policy_is_preserved_and_unknown_requirement_blocks_shortlist(self):
        class StrongBackend(LocalBacktest):
            def run(self, *args):
                return {"rank_ic": 0.2, "rank_ic_days": 18, "minimum_ic_days": 16,
                        "mean_coverage": 1, "net_bps_per_day": 8, "stress_net_bps_per_day": 3}

        class OptimisticCritic(OfflineTeam):
            def choose(self, role, instruction, context, tools, history):
                if role == "Validation & Critic Agent":
                    return "submit_review", {"action": "shortlist", "reason": "Positive results", "cadence": 1}, [], ""
                return super().choose(role, instruction, context, tools, history)

        policy = Policy.parse({"title": "Firm", "new_risk_rule": {"sector_limit": 0.02}})
        report = self.run_team(policy=policy, backend=StrongBackend(), model=OptimisticCritic(), holdout=True, rounds=1)
        self.assertEqual(report["policy"]["new_risk_rule"], {"sector_limit": 0.02})
        self.assertEqual(report["attempts"][0]["decision"], "blocked_shortlist")
        self.assertEqual(report["holdout"]["status"], "no_eligible_candidate")

        class GrossOnly(StrongBackend):
            def capabilities(self):
                return {**super().capabilities(), "net_costs": False}

        report = self.run_team(backend=GrossOnly(), model=OptimisticCritic(), rounds=1)
        self.assertEqual(report["attempts"][0]["decision"], "blocked_shortlist")
        self.assertIn("net cost methodology is unverified", report["attempts"][0]["reasons"])

    def test_model_can_select_tools_and_receives_results_by_call_id(self):
        model = OpenAIToolModel("fake-only", "test")
        calls = [("query", "first", {"query": "volume"}), ("done", "second", {"reason": "checked"})]
        responses = [io.BytesIO(json.dumps({"status": "completed", "output": [
            {"type": "function_call", "name": name, "call_id": call_id, "arguments": json.dumps(args)},
        ]}).encode()) for name, call_id, args in calls]
        tools = {
            "query": Tool("Query", {"query": str}, lambda query: {"match": query}),
            "done": Tool("Complete", {"reason": str}, lambda reason: Finish({"reason": reason})),
        }
        with patch("alpha_lab.agent_runtime.request.urlopen", side_effect=responses) as post:
            result = AgentLoop(model, 2).run("test role", "test instructions", {}, tools)
        self.assertEqual(result, {"reason": "checked"})
        first = json.loads(post.call_args_list[0].args[0].data)
        second = json.loads(post.call_args_list[1].args[0].data)
        self.assertEqual(first["tool_choice"], "required")
        self.assertFalse(first["store"])
        self.assertEqual([tool["name"] for tool in first["tools"]], ["query", "done"])
        self.assertEqual(second["input"][-1]["call_id"], "first")
        self.assertEqual(json.loads(second["input"][-1]["output"]), {"match": "volume"})


if __name__ == "__main__":
    unittest.main()
