"""Persistent experiment history and an atomic local holdout receipt."""
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, sort_keys=True)


class Registry:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, context TEXT NOT NULL, outcome TEXT
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    step INTEGER NOT NULL,
                    scope TEXT NOT NULL,
                    expression TEXT,
                    event TEXT NOT NULL,
                    PRIMARY KEY (run_id, step)
                );
                CREATE INDEX IF NOT EXISTS attempt_scope ON attempts(scope);
                CREATE TABLE IF NOT EXISTS holdouts (
                    data_sha TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    result TEXT
                );
            """)

    def connect(self):
        db = sqlite3.connect(self.path)
        db.execute("PRAGMA foreign_keys = ON")
        return db

    def start(self, run_id, context):
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO runs (id, context) VALUES (?, ?)", (run_id, encode(context)))

    def record(self, run_id, scope, step, expression, event):
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO attempts VALUES (?, ?, ?, ?, ?)", (run_id, step, scope, expression, encode(event)))

    def has_attempt(self, scope, expression):
        with closing(self.connect()) as db:
            return db.execute(
                "SELECT 1 FROM attempts WHERE scope = ? AND expression = ? LIMIT 1",
                (scope, expression),
            ).fetchone() is not None

    def recent(self, scope):
        with closing(self.connect()) as db:
            rows = db.execute(
                "SELECT event FROM attempts WHERE scope = ? AND expression IS NOT NULL ORDER BY rowid DESC LIMIT 12",
                (scope,),
            ).fetchall()
        return [json.loads(event) for (event,) in rows]

    def finish(self, run_id, outcome):
        with closing(self.connect()) as db, db:
            db.execute("UPDATE runs SET outcome = ? WHERE id = ?", (encode(outcome), run_id))

    def open_holdout(self, data_sha, run_id, evaluate_once):
        with closing(self.connect()) as db, db:
            claim = db.execute("INSERT OR IGNORE INTO holdouts VALUES (?, ?, NULL)", (data_sha, run_id))
            if claim.rowcount == 0:
                return {"status": "already_opened"}
            result = evaluate_once()
            db.execute("UPDATE holdouts SET result = ? WHERE data_sha = ?", (encode(result), data_sha))
            return result
