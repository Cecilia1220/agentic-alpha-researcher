"""CSV contract, session clock, and feature registry."""
import csv
import hashlib
import io
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from statistics import fmean


@dataclass(frozen=True)
class Feature:
    description: str
    availability_column: str | None = None


FEATURES = {
    "ret_1d": Feature("One-session close-to-close return, observed at session close."),
    "ret_5d": Feature("Five-session close-to-close return, observed at session close."),
    "volume_shock": Feature("Current volume relative to the prior five-session mean. An activity measure without trade direction."),
    "value_score": Feature("Synthetic report score with a release lag.", "value_available"),
}
EXECUTION_FIELDS = {"open", "forward_return", "target"}


@dataclass(frozen=True)
class Panel:
    dates: tuple[date, ...]
    symbols: tuple[str, ...]
    rows: dict
    sha256: str
    schema: str = "features"
    registry: dict[str, Feature] = field(default_factory=lambda: FEATURES.copy())

    def forward_return(self, session: int, symbol: str) -> float:
        entry, exit_ = self.dates[session + 1:session + 3]
        return self.rows[exit_, symbol]["open"] / self.rows[entry, symbol]["open"] - 1

    def feature(self, name: str, observation: int, decision: int, symbol: str) -> float | None:
        """Read an observation only if released by the decision session's close."""
        if not 0 <= observation <= decision:
            return None
        spec = self.registry[name]
        row = self.rows[self.dates[observation], symbol]
        if row[name] is not None and spec.availability_column and row[spec.availability_column] > self.dates[decision]:
            return None
        return row[name]

    def explore(self, end: int, allowed=None) -> dict:
        """Summaries stop at the development boundary and exclude targets."""
        summaries = {}
        for name in self.registry if allowed is None else allowed:
            feature = self.registry[name]
            values = [
                self.feature(name, session, session, symbol)
                for session in range(end) for symbol in self.symbols
            ]
            observed = [value for value in values if value is not None]
            summaries[name] = {
                "description": feature.description,
                "decision_day_coverage": round(len(observed) / len(values), 4),
                "availability_column": feature.availability_column,
                "mean_absolute_value": round(fmean(abs(x) for x in observed), 6) if observed else None,
            }
        return summaries


def load_panel(path: Path) -> Panel:
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    base = {"date", "symbol", "open", "value_available", "value_score"}
    raw_fields = base | {"close", "volume"}
    feature_fields = base | set(FEATURES)
    fields = set(reader.fieldnames or ())
    if fields not in (raw_fields, feature_fields) or len(reader.fieldnames) != len(fields):
        raise ValueError("CSV must supply close and volume or the registered precomputed features")
    is_raw = fields == raw_fields
    rows = {}
    for line, record in enumerate(reader, 2):
        try:
            if None in record or any(value is None for value in record.values()):
                raise ValueError("incorrect field count")
            session = date.fromisoformat(record["date"])
            symbol = record["symbol"].strip()
            release = date.fromisoformat(record["value_available"])
            if not symbol or (session, symbol) in rows or release < session:
                raise ValueError("blank symbol, duplicate key, or release before observation")
            open_price = float(record["open"])
            if not math.isfinite(open_price) or not 0.001 <= open_price <= 1_000_000:
                raise ValueError("open price outside supported bounds")
            row = {"open": open_price, "value_available": release}
            for feature in (("value_score", "close", "volume") if is_raw else FEATURES):
                value = float(record[feature]) if record[feature].strip() else None
                if value is not None and (not math.isfinite(value) or abs(value) > 1_000_000_000_000):
                    raise ValueError(f"{feature} outside supported bounds")
                row[feature] = value
            if is_raw and (row["close"] is None or row["close"] < 0.001
                           or row["volume"] is None or row["volume"] < 0.000001):
                raise ValueError("raw close and volume must be positive")
            rows[session, symbol] = row
        except ValueError as exc:
            raise ValueError(f"CSV row {line}: {exc}") from exc
    dates = tuple(sorted({session for session, _ in rows}))
    symbols = tuple(sorted({symbol for _, symbol in rows}))
    if len(dates) < 40 or len(symbols) < 6 or len(rows) != len(dates) * len(symbols):
        raise ValueError("CSV needs a rectangular panel of at least 40 sessions and six symbols")
    if is_raw:
        for symbol in symbols:
            history = [rows[session, symbol] for session in dates]
            for i, row in enumerate(history):
                for lag in (1, 5):
                    row[f"ret_{lag}d"] = row["close"] / history[i - lag]["close"] - 1 if i >= lag else None
                row["volume_shock"] = (
                    row["volume"] / fmean(item["volume"] for item in history[i - 5:i]) - 1
                    if i >= 5 else None
                )
    return Panel(dates, symbols, rows, hashlib.sha256(raw).hexdigest(), "raw" if is_raw else "features")
