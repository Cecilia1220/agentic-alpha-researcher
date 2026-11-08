"""Keep firm documents intact and apply only supported, deterministic controls."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

CONTROLS = {"asset_class", "allowed_datasets", "allowed_features", "minimum_cost_bps", "allow_models"}
DOCUMENT_FIELDS = {"title", "version", "description", "notes", "controls"}


@dataclass(frozen=True)
class Policy:
    document: object
    unresolved: tuple[str, ...]
    allowed_datasets: tuple[str, ...] | None
    allowed_features: tuple[str, ...] | None
    minimum_cost_bps: float
    allow_models: bool

    @classmethod
    def load(cls, path: Path | None):
        if path is None:
            return cls.parse({})
        if path.stat().st_size > 24_000:
            raise ValueError("policy document exceeds the local review budget")
        raw = path.read_text(encoding="utf-8")
        return cls.parse(json.loads(raw) if path.suffix.lower() == ".json" else raw)

    @classmethod
    def parse(cls, document):
        if not isinstance(document, dict):
            return cls(document, ("policy document requires a verified control mapping",), None, None, 0, True)
        fields = dict(document)
        nested = fields.pop("controls", {})
        if not isinstance(nested, dict) or any(key in fields for key in nested):
            raise ValueError("controls must be a JSON object without duplicate top-level settings")
        fields.update(nested)
        unresolved = set(fields) - CONTROLS - DOCUMENT_FIELDS
        for name in ("description", "notes"):
            if fields.get(name):
                unresolved.add(f"review free-text {name} for material constraints")
        if fields.get("asset_class", "daily_equities") != "daily_equities":
            raise ValueError("this research harness currently executes daily equity strategies only")

        def allowlist(name):
            value = fields.get(name)
            if value is None:
                return None
            if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
                raise ValueError(f"{name} must be a list of names")
            return tuple(value)

        cost, models = fields.get("minimum_cost_bps", 0), fields.get("allow_models", True)
        if type(cost) not in (int, float) or not math.isfinite(cost) or not 0 <= cost <= 100:
            raise ValueError("policy minimum cost must be within 0–100 bps")
        if type(models) is not bool:
            raise ValueError("allow_models must be boolean")
        return cls(document, tuple(sorted(unresolved)), allowlist("allowed_datasets"),
                   allowlist("allowed_features"), cost, models)

    def permits_dataset(self, name):
        return self.allowed_datasets is None or name in self.allowed_datasets
