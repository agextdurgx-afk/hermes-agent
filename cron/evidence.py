"""Retention pins for immutable evidence referenced by durable cron receipts.

The receipt writer inserts its exact pin inventory in its existing transaction.
Retention consumers are read-only and never release or rewrite these pins.
"""
from __future__ import annotations
import re
import sqlite3
from pathlib import Path
from contextlib import contextmanager


def evidence_pins(conn, *, kind=None, authority_id=None):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cron_evidence_pins'").fetchone():
        return []
    rows = []
    for authority, entry_kind, identity, digest in conn.execute(
            "SELECT authority_id,kind,identity,sha256 FROM cron_evidence_pins ORDER BY authority_id,kind,identity"):
        if entry_kind not in ("execution", "log") or not authority or not identity or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("cron evidence retention pin is malformed")
        if (kind is None or kind == entry_kind) and (authority_id is None or authority_id == authority):
            rows.append({"authority_id": authority, "kind": entry_kind, "identity": identity, "sha256": digest})
    return rows


def retain_evidence(conn, authority_id, entries):
    """Called only inside the receipt owner's transaction; never commits."""
    conn.execute("CREATE TABLE IF NOT EXISTS cron_evidence_pins (authority_id TEXT NOT NULL,kind TEXT NOT NULL,identity TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(authority_id,kind,identity))")
    expected = sorted([{"authority_id": authority_id, **entry} for entry in entries], key=lambda row: (row["kind"], row["identity"]))
    if not authority_id or not expected or len({(row["kind"], row["identity"]) for row in expected}) != len(expected):
        raise ValueError("cron evidence retention inventory is empty or duplicated")
    current = evidence_pins(conn, authority_id=authority_id)
    if current:
        if current != expected:
            raise ValueError("cron evidence retention authority already has different pins")
        return
    for row in expected:
        if row["kind"] not in ("execution", "log") or not row["identity"] or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("cron evidence retention pin is malformed")
        conn.execute("INSERT INTO cron_evidence_pins VALUES (?,?,?,?)", (authority_id, row["kind"], row["identity"], row["sha256"]))


def unresolved_incident_evidence(conn):
    """Retention before review: unresolved incidents still need their evidence.

    This is a live reference inventory, not a receipt or an approval. A later
    closed incident releases only this guard; immutable receipt pins still win.
    """
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cron_incidents'").fetchone():
        return {"job_ids": set(), "log_paths": set()}
    jobs, paths = set(), set()
    for job, output in conn.execute("SELECT job_id,output_file FROM cron_incidents WHERE state!='closed'"):
        if not isinstance(job, str) or not job:
            raise ValueError("unresolved cron incident has no job identity")
        jobs.add(job)
        if output is not None:
            if not isinstance(output, str) or not output:
                raise ValueError("unresolved cron incident has malformed output identity")
            paths.add(output)
    return {"job_ids": jobs, "log_paths": paths}


def retained_log_paths(database_path):
    """An absent database has no pins; unreadable state must stop pruning."""
    path = Path(database_path)
    try:
        path.stat()
    except FileNotFoundError:
        return set()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.execute("BEGIN")
        return {row["identity"] for row in evidence_pins(conn, kind="log")} | unresolved_incident_evidence(conn)["log_paths"]
    finally:
        conn.rollback()
        conn.close()


@contextmanager
def log_retention_guard(database_path):
    """Keep pin writers excluded from inspection through filesystem deletion.

    A read-only pin query followed by unlink has a race with a receipt writer.
    The same immediate SQLite transaction used by receipt writers serializes
    that whole interval. An empty database is harmless when cron has not yet
    recorded an execution; creating it also closes the absent-database race.
    """
    conn = sqlite3.connect(Path(database_path), timeout=30)
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield {row["identity"] for row in evidence_pins(conn, kind="log")} | unresolved_incident_evidence(conn)["log_paths"]
    finally:
        conn.rollback()
        conn.close()
