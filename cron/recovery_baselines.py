"""Prospective recovery checkpoints. Historical incidents remain unresolved.

Registration is conditional on an exact incident/failure snapshot and a drained
job. It never acknowledges, closes, updates or deletes an incident. A later
failure or changed historical row invalidates the checkpoint; no error-prefix
or timestamp-tolerance matching is used.
"""
from __future__ import annotations
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from cron import incidents
from cron.evidence import evidence_pins, retain_evidence


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _snapshot(conn, job_id, *, require_drained=True):
    rows = [dict(row) for row in conn.execute("SELECT * FROM cron_incidents WHERE state!='closed' ORDER BY id")]
    if not rows or any(row["job_id"] != job_id for row in rows):
        raise ValueError("baseline requires a nonempty exact single-job incident inventory")
    if conn.execute("SELECT 1 FROM executions WHERE job_id=? AND status='unknown' LIMIT 1", (job_id,)).fetchone():
        raise ValueError("baseline job has an unknown execution outcome")
    if require_drained and conn.execute("SELECT 1 FROM executions WHERE job_id=? AND status IN ('claimed','running','unknown') LIMIT 1", (job_id,)).fetchone():
        raise ValueError("baseline job is not proven drained")
    failures = []
    errors = {}
    for row in conn.execute("SELECT * FROM executions WHERE job_id=? AND status='failed' ORDER BY id", (job_id,)):
        item = dict(row)
        error = item.pop("error") or ""
        if not item.get("finished_at"):
            raise ValueError("baseline failed execution is not terminal")
        errors[item["id"]] = error
        item["error_sha256"] = hashlib.sha256(error.encode()).hexdigest()
        failures.append(item)
    logs = []
    for row in rows:
        path = Path(row["output_file"])
        content = path.read_bytes()
        text = content.decode()
        start = text.find("\n\nScript exited with code ")
        if start < 0 or f"**Job ID:** {job_id}\n" not in text[:start+2]:
            raise ValueError("baseline incident log lacks exact job identity")
        error = text[start+2:].removesuffix("\n")
        matches = [key for key, value in errors.items() if value == error]
        if not matches or row["error"] != error[:500]:
            raise ValueError("baseline incident is not bound to a complete recorded failure")
        logs.append({"incident_id": row["id"], "path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
                     "bytes": len(content), "matching_execution_ids": matches})
    return {"schema_version": 1, "job_id": job_id, "incidents": rows, "failed_executions": failures, "logs": logs}


def capture_recovery_snapshot(job_id, *, require_drained=True):
    conn = sqlite3.connect(f"file:{incidents._db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN")
        return _snapshot(conn, job_id, require_drained=require_drained)
    finally:
        conn.rollback(); conn.close()


def inspect_recovery_snapshot(job_id):
    """Observe exact history while a harmless scheduled tick may be active."""
    return capture_recovery_snapshot(job_id, require_drained=False)


def _verify_authority(request):
    authority = request.get("authority", {})
    for name in ("baseline", "review"):
        proof = authority.get(name, {})
        content = Path(proof["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != proof.get("sha256"):
            raise ValueError("baseline authority artifact changed")
    baseline = json.loads(Path(authority["baseline"]["path"]).read_bytes())
    review = json.loads(Path(authority["review"]["path"]).read_bytes())
    if (baseline.get("mode") != "prospective_recovery_baseline_v1"
            or any(not re.fullmatch(r"[0-9a-f]{40}", str(baseline.get(key, ""))) for key in ("reviewed_tezoff_sha", "reviewed_hermes_sha"))
            or baseline.get("incident_disposition") != "preserved_unresolved"
            or baseline.get("cron_snapshot") != request.get("snapshot")
            or review.get("recovery_baseline_sha256") != authority["baseline"]["sha256"]
            or review.get("incident_disposition") != "preserved_unresolved"
            or review.get("verdict") != "pass"
            or review.get("reviewer_model") != "gpt-6-astra"
            or review.get("schema_version") != 5 or review.get("review_version") != 5
            or review.get("replacement_authorized") is not True
            or any(review.get("controls", {}).get(f"A{i}") != "pass" for i in range(1, 13))
            or review.get("reviewed_tezoff_sha") != baseline.get("reviewed_tezoff_sha")
            or review.get("reviewed_hermes_sha") != baseline.get("reviewed_hermes_sha")):
        raise ValueError("baseline lacks exact prospective authority")


def _retention_entries(request):
    snapshot = request["snapshot"]
    entries = [{"kind": "execution", "identity": row["id"], "sha256": _digest(row)} for row in snapshot["failed_executions"]]
    logs = {row["path"]: row["sha256"] for row in snapshot["logs"]}
    return sorted(entries + [{"kind": "log", "identity": path, "sha256": digest} for path, digest in logs.items()], key=lambda row: (row["kind"], row["identity"]))


def _verify_retention(conn, request):
    expected = [{"authority_id": request["baseline_id"], **entry} for entry in _retention_entries(request)]
    if evidence_pins(conn, authority_id=request["baseline_id"]) != expected:
        raise ValueError("baseline retention evidence changed")


def register_recovery_baseline(request):
    """One exact write transaction; lost-response retries return one receipt."""
    if request.get("schema_version") != 1 or request.get("baseline_id") != _digest({"snapshot": request.get("snapshot"), "authority": request.get("authority")}):
        raise ValueError("baseline request identity changed")
    job_id = request["snapshot"]["job_id"]
    with incidents._transaction() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS cron_recovery_baselines (id TEXT PRIMARY KEY, job_id TEXT NOT NULL UNIQUE, request_json TEXT NOT NULL, receipt_json TEXT NOT NULL)")
        conn.commit(); conn.execute("BEGIN IMMEDIATE")
        _verify_authority(request)
        if _snapshot(conn, job_id) != request["snapshot"]:
            raise ValueError("baseline incident or failure inventory changed")
        prior = conn.execute("SELECT * FROM cron_recovery_baselines WHERE job_id=?", (job_id,)).fetchone()
        if prior:
            if prior["id"] != request["baseline_id"] or json.loads(prior["request_json"]) != request:
                raise ValueError("a recovery baseline already exists for this job")
            _verify_retention(conn, request)
            return json.loads(prior["receipt_json"])
        receipt = {"schema_version": 1, "baseline_id": request["baseline_id"], "incident_disposition": "preserved_unresolved",
                   "request_sha256": _digest(request), "registered_at": incidents._hermes_now().isoformat()}
        retain_evidence(conn, request["baseline_id"], _retention_entries(request))
        conn.execute("INSERT INTO cron_recovery_baselines VALUES (?,?,?,?)", (request["baseline_id"], job_id, json.dumps(request,sort_keys=True), json.dumps(receipt,sort_keys=True)))
        return receipt


def inspect_recovery_baselines():
    """Read-only. A drifted checkpoint is an error, never an absent record."""
    conn = sqlite3.connect(f"file:{incidents._db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN")
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cron_recovery_baselines'").fetchone():
            return []
        output = []
        for row in conn.execute("SELECT * FROM cron_recovery_baselines ORDER BY id"):
            request = json.loads(row["request_json"])
            receipt = json.loads(row["receipt_json"])
            if (receipt.get("schema_version") != 1 or receipt.get("incident_disposition") != "preserved_unresolved"
                    or receipt["baseline_id"] != row["id"] or request.get("baseline_id") != row["id"]
                    or row["id"] != _digest({"snapshot": request.get("snapshot"), "authority": request.get("authority")})
                    or row["job_id"] != request.get("snapshot", {}).get("job_id")
                    or receipt["request_sha256"] != _digest(request)):
                raise ValueError("stored baseline receipt changed")
            _verify_authority(request)
            _verify_retention(conn, request)
            if _snapshot(conn, row["job_id"], require_drained=False) != request["snapshot"]:
                raise ValueError("preserved incident or a later execution changed")
            output.append({"request": request, "receipt": receipt})
        return output
    finally:
        conn.rollback(); conn.close()
