"""Optional local transcripts, scoped to backend URL and dataset ID."""

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path


def connection():
    folder = (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        / "rag-demo-system"
    )
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    file = folder / "sessions.sqlite3"
    file.touch(mode=0o600, exist_ok=True)
    db = sqlite3.connect(file)
    db.execute(
        "CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, endpoint TEXT, dataset TEXT, title TEXT, transcript TEXT)"
    )
    return db


def save(scope, title, transcript):
    with closing(connection()) as db, db:
        db.execute(
            "INSERT INTO sessions(endpoint,dataset,title,transcript) VALUES (?,?,?,?)",
            (*scope, title, json.dumps(transcript, ensure_ascii=False)),
        )


def list_saved(scope):
    with closing(connection()) as db:
        return db.execute(
            "SELECT title,id FROM sessions WHERE endpoint=? AND dataset=? ORDER BY id DESC",
            scope,
        ).fetchall()


def read(scope, ident):
    with closing(connection()) as db:
        row = db.execute(
            "SELECT transcript FROM sessions WHERE endpoint=? AND dataset=? AND id=?",
            (*scope, ident),
        ).fetchone()
        return json.loads(row[0]) if row else []


def delete(scope, ident=None):
    with closing(connection()) as db, db:
        if ident is None:
            db.execute("DELETE FROM sessions WHERE endpoint=? AND dataset=?", scope)
        else:
            db.execute(
                "DELETE FROM sessions WHERE endpoint=? AND dataset=? AND id=?",
                (*scope, ident),
            )
