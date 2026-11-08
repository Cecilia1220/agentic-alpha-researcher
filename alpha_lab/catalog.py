"""Discover data by metadata and normalize approved sources into daily equity panels."""
import csv
import hashlib
import json
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import date
from pathlib import Path
from statistics import fmean

from .data import EXECUTION_FIELDS, Feature, Panel, load_panel
from .literature import terms

MAX_ROWS = 250_000
MAX_BYTES = 32 * 1024 * 1024


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _number(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected a numeric observation") from exc
    if type(value) is bool or not math.isfinite(number) or abs(number) > 1e12:
        raise ValueError("invalid numeric observation")
    return number


def _rows(source, path):
    fmt = source["format"]
    if fmt == "sqlite":
        table = source.get("table", "")
        if not isinstance(table, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", table):
            raise ValueError("SQLite source needs a catalog-approved table name")
        try:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
                db.row_factory = sqlite3.Row
                records = [dict(row) for row in db.execute(f'SELECT * FROM "{table}" LIMIT ?', (MAX_ROWS + 1,))]
        except sqlite3.Error as exc:
            raise ValueError("unable to read the registered SQLite table") from exc
    else:
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("source exceeds the local file budget")
        with path.open(encoding="utf-8", newline="") as stream:
            if fmt == "csv":
                reader = csv.DictReader(stream)
                if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                    raise ValueError("invalid CSV header")
                records = []
                for row in reader:
                    records.append(row)
                    if len(records) > MAX_ROWS:
                        break
            elif fmt == "json":
                records = json.load(stream)
            else:
                raise ValueError("register a CSV, JSON, or read-only SQLite adapter")
    if (not isinstance(records, list) or not records or len(records) > MAX_ROWS
            or any(not isinstance(row, dict) or None in row for row in records)):
        raise ValueError("expected a nonempty, bounded collection of records")
    return records


def _normalize(source, records):
    keys = source["keys"]
    specs = source["features"]
    if not isinstance(keys, dict) or not all(isinstance(keys.get(k), str) for k in ("date", "symbol", "open")):
        raise ValueError("catalog must map session, security and execution open")
    if not isinstance(specs, dict) or not specs or len(specs) > 100:
        raise ValueError("catalog needs documented features")
    features, required = {}, set(keys.values())
    for name, spec in specs.items():
        if (not isinstance(name, str) or not name.isidentifier() or name in EXECUTION_FIELDS
                or not isinstance(spec, dict) or not isinstance(spec.get("column"), str)
                or not isinstance(spec.get("description"), str)
                or spec.get("availability") not in ("close", "release_date")):
            raise ValueError("feature must document its source and availability convention")
        if spec["column"] in keys.values():
            raise ValueError("execution and key columns cannot be signal features")
        required.add(spec["column"])
        release_key = None
        if spec["availability"] == "release_date":
            release = spec.get("release_column")
            if not isinstance(release, str):
                raise ValueError("delayed feature needs a release-date column")
            required.add(release)
            release_key = f"@release:{name}"
        features[name] = Feature(spec["description"][:300], release_key)

    rows = {}
    for index, raw in enumerate(records, 1):
        if not required <= raw.keys():
            raise ValueError(f"source row {index} is missing a mapped column")
        try:
            session = date.fromisoformat(raw[keys["date"]])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"source row {index} has an invalid session") from exc
        symbol = raw[keys["symbol"]]
        if not isinstance(symbol, str) or not symbol.strip() or (session, symbol) in rows:
            raise ValueError(f"source row {index} has an invalid or duplicate security key")
        price = _number(raw[keys["open"]])
        if price is None or not 0.001 <= price <= 1e6:
            raise ValueError(f"source row {index} has an invalid execution price")
        row = {"open": price}
        for name, feature in features.items():
            spec = specs[name]
            row[name] = _number(raw[spec["column"]])
            if feature.availability_column and row[name] is not None:
                try:
                    release = date.fromisoformat(raw[spec["release_column"]])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"source row {index} has no valid release date for {name}") from exc
                if release < session:
                    raise ValueError(f"source row {index} releases {name} before its observation")
                row[feature.availability_column] = release
        rows[session, symbol] = row
    dates = tuple(sorted({session for session, _ in rows}))
    symbols = tuple(sorted({symbol for _, symbol in rows}))
    if len(dates) < 40 or len(symbols) < 6 or len(rows) != len(dates) * len(symbols):
        raise ValueError("local evaluator needs a complete execution panel of at least 40 sessions and six securities")
    return Panel(dates, symbols, rows, _digest([source, records]), source["format"], features)


class Catalog:
    def __init__(self, entries, root):
        if not isinstance(entries, list) or not entries:
            raise ValueError("catalog needs at least one source")
        self.entries = {}
        self.root = root.resolve()
        self.panels = {}
        for entry in entries:
            if (not isinstance(entry, dict) or any(
                not isinstance(entry.get(key), str) or not entry[key].strip()
                for key in ("id", "description", "format", "path")
            ) or entry["id"] in self.entries):
                raise ValueError("catalog sources need unique IDs, descriptions, formats and paths")
            if entry["format"] != "legacy_csv" and not {"keys", "features"} <= entry.keys():
                raise ValueError("catalog source needs key and feature mappings")
            self.entries[entry["id"]] = entry

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(path.read_text(encoding="utf-8")), path.parent)

    def search(self, query, permits):
        query_terms = terms(query)
        matches = []
        for entry in self.entries.values():
            if not permits(entry["id"]):
                continue
            features = entry.get("features", {})
            names = [key for key in features if isinstance(key, str)] if isinstance(features, dict) else []
            document = " ".join([entry["id"], entry["description"], *names])
            relevance = len(query_terms & terms(document))
            matches.append((relevance, {
                "id": entry["id"], "description": entry["description"][:400], "format": entry["format"],
            }))
        return [item for _, item in sorted(matches, key=lambda pair: -pair[0])[:12]]

    def load(self, source_id):
        if source_id not in self.entries:
            raise ValueError("unknown catalog source")
        if source_id not in self.panels:
            entry = self.entries[source_id]
            path = (self.root / entry["path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("catalog source must be within the catalog directory")
            if entry["format"] == "legacy_csv":
                panel = load_panel(path)
            else:
                panel = _normalize(entry, _rows(entry, path))
            self.panels[source_id] = panel
        return self.panels[source_id]


def derive(panel, output, source, operation, window):
    """Materialize one causal recipe using only data available at each close."""
    if (not output.isidentifier() or output in panel.registry or output in EXECUTION_FIELDS
            or source not in panel.registry or not 1 <= window <= 20
            or operation not in ("change", "relative_to_mean", "carry_forward")):
        raise ValueError("invalid recipe; use a new name and a documented feature with a lookback of 1–20")
    rows = {}
    for t, session in enumerate(panel.dates):
        for symbol in panel.symbols:
            values = [panel.feature(source, day, t, symbol) for day in range(t - window, t + 1)]
            value = None
            if operation == "carry_forward":
                value = next((item for item in reversed(values) if item is not None), None)
            elif None not in values:
                base = values[0] if operation == "change" else fmean(values[:-1])
                if base and abs(base) >= 1e-12:
                    ratio = values[-1] / base - 1
                    if abs(ratio) <= 1e12:
                        value = ratio
            rows[session, symbol] = {**panel.rows[session, symbol], output: value}
    description = f"{operation} of {source} using {window} prior sessions; known by decision close."
    return replace(panel, rows=rows, registry={**panel.registry, output: Feature(description)},
                   sha256=_digest([panel.sha256, output, source, operation, window]))
