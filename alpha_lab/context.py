"""Research brief, firm rules, and time-stamped background inputs."""
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .data import FEATURES


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Brief:
    objective: str
    discovery: str = "literature_first"
    allowed_features: tuple[str, ...] = tuple(FEATURES)

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or not value.keys() <= {"objective", "discovery", "allowed_features"}:
            raise ValueError("brief fields are objective, discovery and allowed_features")
        objective = value.get("objective")
        discovery = value.get("discovery", "literature_first")
        if not isinstance(objective, str) or not objective.strip() or len(objective) > 400:
            raise ValueError("brief objective must contain 1–400 characters")
        if discovery not in ("literature_first", "data_first"):
            raise ValueError("discovery must be literature_first or data_first")
        return cls(objective, discovery, _feature_list(value.get("allowed_features", list(FEATURES)), registered=False))


@dataclass(frozen=True)
class FirmPolicy:
    allowed_features: tuple[str, ...] = tuple(FEATURES)
    minimum_cost_bps: float = 0.0
    allow_models: bool = True

    @classmethod
    def parse(cls, value):
        keys = {"asset_class", "allowed_features", "minimum_cost_bps", "allow_models"}
        if not isinstance(value, dict) or not value.keys() <= keys:
            raise ValueError("unknown firm policy fields")
        if value.get("asset_class", "daily_equities") != "daily_equities":
            raise ValueError("this evaluator supports daily equities only")
        features = _feature_list(value.get("allowed_features", list(FEATURES)))
        cost = value.get("minimum_cost_bps", 0.0)
        if type(cost) not in (int, float) or not math.isfinite(cost) or not 0 <= cost <= 100:
            raise ValueError("firm minimum cost must be between 0 and 100 bps")
        models = value.get("allow_models", True)
        if type(models) is not bool:
            raise ValueError("allow_models must be a boolean")
        return cls(features, cost, models)


def _feature_list(features, *, registered=True):
    if (not isinstance(features, list) or not features
            or any(not isinstance(name, str) or not name.isidentifier()
                   or (registered and name not in FEATURES) for name in features)):
        raise ValueError("allowed_features needs at least one registered feature")
    return tuple(dict.fromkeys(features))


def market_snapshot(records, as_of):
    """Use the most recent background snapshot known at the start of the research window."""
    if not isinstance(records, list):
        raise ValueError("market context must be a list of dated snapshots")
    eligible = []
    for item in records:
        if (not isinstance(item, dict) or item.keys() != {"available_on", "source", "summary"}
                or any(not isinstance(s, str) or not s.strip() or len(s) > 500 for s in item.values())):
            raise ValueError("each market snapshot needs available_on, source and summary")
        published = date.fromisoformat(item["available_on"])
        if published < as_of:
            eligible.append((published, item))
    return max(eligible, key=lambda pair: pair[0])[1] if eligible else None
