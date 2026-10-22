"""SQLite persistence of a finished memory run (records, operation log, resolution log,
size trace, metadata). One file per run: runs/<video_id>/<run>/memory.sqlite."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memvid.memory.records import RecordT
from memvid.memory.store import parse_record

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE records (pos INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
                      line TEXT NOT NULL, json TEXT NOT NULL);
CREATE TABLE ops (seq INTEGER PRIMARY KEY, chunk_id INTEGER NOT NULL, op TEXT NOT NULL,
                  label TEXT NOT NULL, record_id TEXT NOT NULL, record TEXT, reason TEXT);
CREATE INDEX ops_chunk ON ops(chunk_id);
CREATE INDEX ops_record ON ops(record_id);
CREATE TABLE resolution (pos INTEGER PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE sizes (chunk_id INTEGER PRIMARY KEY, tokens INTEGER NOT NULL, n_records INTEGER NOT NULL);
CREATE TABLE bounds (chunk_id INTEGER PRIMARY KEY, t_start REAL NOT NULL, t_end REAL NOT NULL);
"""


@dataclass
class SavedMemory:
    meta: dict[str, Any]
    records: dict[str, RecordT]
    ops: list[dict[str, Any]]
    resolution: list[dict[str, Any]]
    sizes: list[dict[str, Any]]
    bounds: dict[int, tuple[float, float]]


def save_memory(path: Path, *, meta: dict[str, Any], records: dict[str, RecordT],
                ops: list[dict[str, Any]], resolution: list[dict[str, Any]],
                sizes: list[dict[str, Any]], bounds: dict[int, tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.sqlite")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO meta VALUES (?, ?)",
                        [(k, json.dumps(v, default=str)) for k, v in meta.items()])
        con.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                        [(i, r.id, r.kind, r.line(), r.model_dump_json())
                         for i, r in enumerate(records.values())])
        con.executemany("INSERT INTO ops VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [(o["seq"], o["chunk_id"], o["op"], o["label"], o["record_id"],
                          json.dumps(o["record"]) if o["record"] is not None else None, o["reason"])
                         for o in ops])
        con.executemany("INSERT INTO resolution VALUES (?, ?)",
                        [(i, json.dumps(r)) for i, r in enumerate(resolution)])
        con.executemany("INSERT INTO sizes VALUES (?, ?, ?)",
                        [(s["chunk_id"], s["tokens"], s["n_records"]) for s in sizes])
        con.executemany("INSERT INTO bounds VALUES (?, ?, ?)",
                        [(c, a, b) for c, (a, b) in sorted(bounds.items())])
        con.commit()
    finally:
        con.close()
    tmp.replace(path)


def load_memory(path: Path) -> SavedMemory:
    if not path.is_file():
        raise FileNotFoundError(path)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        meta = {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM meta")}
        records = {}
        for (js,) in con.execute("SELECT json FROM records ORDER BY pos"):
            r = parse_record(json.loads(js))
            records[r.id] = r
        ops = [{"seq": s, "chunk_id": c, "op": o, "label": lb, "record_id": rid,
                "record": json.loads(rec) if rec is not None else None, "reason": reason}
               for s, c, o, lb, rid, rec, reason in con.execute("SELECT * FROM ops ORDER BY seq")]
        resolution = [json.loads(js) for (js,) in con.execute("SELECT json FROM resolution ORDER BY pos")]
        sizes = [{"chunk_id": c, "tokens": t, "n_records": n}
                 for c, t, n in con.execute("SELECT * FROM sizes ORDER BY chunk_id")]
        bounds = {c: (a, b) for c, a, b in con.execute("SELECT * FROM bounds")}
    finally:
        con.close()
    return SavedMemory(meta=meta, records=records, ops=ops, resolution=resolution, sizes=sizes,
                       bounds=bounds)
