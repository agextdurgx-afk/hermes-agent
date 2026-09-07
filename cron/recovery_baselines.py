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


def _job_scope(job_id, job_ids=None):
    if job_ids is None:
        return [job_id]
    if (not isinstance(job_ids, list) or not 1 <= len(job_ids) <= 64
            or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) for value in job_ids)
            or job_ids != sorted(set(job_ids)) or job_id not in job_ids):
        raise ValueError("operational job scope is not exact and bounded")
    return job_ids


def _snapshot(conn, job_id, *, require_drained=True, job_ids=None):
    scope = _job_scope(job_id, job_ids)
    placeholders = ",".join("?" for _ in scope)
    rows = [dict(row) for row in conn.execute("SELECT * FROM cron_incidents WHERE state!='closed' ORDER BY id")]
    if not rows or any(row["job_id"] not in scope for row in rows):
        raise ValueError("baseline requires a nonempty exact single-job incident inventory")
    if conn.execute(f"SELECT 1 FROM executions WHERE job_id IN ({placeholders}) AND status='unknown' LIMIT 1", scope).fetchone():
        raise ValueError("baseline job has an unknown execution outcome")
    if require_drained and conn.execute(f"SELECT 1 FROM executions WHERE job_id IN ({placeholders}) AND status IN ('claimed','running','unknown') LIMIT 1", scope).fetchone():
        raise ValueError("baseline job is not proven drained")
    failures = []
    errors = {}
    for row in conn.execute(f"SELECT * FROM executions WHERE job_id IN ({placeholders}) AND status='failed' ORDER BY id", scope):
        item = dict(row)
        error = item.pop("error") or ""
        if not item.get("finished_at"):
            raise ValueError("baseline failed execution is not terminal")
        errors[item["id"]] = (item["job_id"], error)
        item["error_sha256"] = hashlib.sha256(error.encode()).hexdigest()
        failures.append(item)
    logs = []
    for row in rows:
        path = Path(row["output_file"])
        content = path.read_bytes()
        text = content.decode()
        start = text.find("\n\nScript exited with code ")
        if start < 0 or f"**Job ID:** {row['job_id']}\n" not in text[:start+2]:
            raise ValueError("baseline incident log lacks exact job identity")
        error = text[start+2:].removesuffix("\n")
        matches = [key for key, value in errors.items() if value == (row["job_id"], error)]
        if not matches or row["error"] != error[:500]:
            raise ValueError("baseline incident is not bound to a complete recorded failure")
        logs.append({"incident_id": row["id"], "path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
                     "bytes": len(content), "matching_execution_ids": matches})
    snapshot = {"schema_version": 1, "job_id": job_id, "incidents": rows, "failed_executions": failures, "logs": logs}
    if job_ids is not None:
        snapshot.update(schema_version=2, job_ids=scope)
    return snapshot


def capture_recovery_snapshot(job_id, *, require_drained=True):
    job_ids = None
    if isinstance(job_id, dict):
        if set(job_id) != {"job_id", "job_ids"}:
            raise ValueError("unknown operational snapshot selector")
        job_id, job_ids = job_id["job_id"], job_id["job_ids"]
    conn = sqlite3.connect(f"file:{incidents._db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN")
        return _snapshot(conn, job_id, require_drained=require_drained, job_ids=job_ids)
    finally:
        conn.rollback(); conn.close()


def inspect_recovery_snapshot(job_id):
    """Observe exact history while a harmless scheduled tick may be active."""
    return capture_recovery_snapshot(job_id, require_drained=False)


def _preparation_request(request):
    if (request.get("schema_version") != 1 or not request.get("job_id")
            or not isinstance(request.get("binding"), dict) or not request["binding"]
            or request.get("preparation_id") != _digest({k: v for k, v in request.items() if k != "preparation_id"})):
        raise ValueError("recovery evidence preparation identity changed")
    scope = request["binding"].get("operational_job_ids")
    if scope is not None:
        _job_scope(request["job_id"], scope)
        if request["binding"].get("authority_scope") != "operational_health_only":
            raise ValueError("scoped evidence requires health-only authority")


def _prepared_snapshot(conn, request, *, current=True):
    _preparation_request(request)
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cron_recovery_preparations'").fetchone():
        if evidence_pins(conn, authority_id="preparation:" + request["preparation_id"]):
            raise ValueError("prepared recovery observation is missing")
        return None
    row = conn.execute("SELECT request_json,snapshot_json FROM cron_recovery_preparations WHERE id=?", (request["preparation_id"],)).fetchone()
    if row is None:
        if evidence_pins(conn, authority_id="preparation:" + request["preparation_id"]):
            raise ValueError("prepared recovery observation is missing")
        return None
    if json.loads(row[0]) != request:
        raise ValueError("prepared recovery evidence request changed")
    snapshot = json.loads(row[1])
    _verify_retention(conn, {"baseline_id": "preparation:" + request["preparation_id"], "snapshot": snapshot})
    if current and snapshot != _snapshot(conn, request["job_id"], require_drained=False, job_ids=request["binding"].get("operational_job_ids")):
        raise ValueError("prepared recovery evidence changed")
    return snapshot


def prepare_recovery_snapshot(request):
    """Atomically capture and retain evidence before review, without authority.

    The caller journals this exact request before invoking us. Capture, pins,
    and the immutable observation commit together; a lost-response retry reads
    the same observation. No baseline is registered and no incident is closed.
    """
    _preparation_request(request)
    with incidents._transaction() as conn:
        conn.commit(); conn.execute("BEGIN IMMEDIATE")
        prior = _prepared_snapshot(conn, request)
        if prior is not None:
            return prior
        snapshot = _snapshot(conn, request["job_id"], job_ids=request["binding"].get("operational_job_ids"))
        conn.execute("CREATE TABLE IF NOT EXISTS cron_recovery_preparations (id TEXT PRIMARY KEY, request_json TEXT NOT NULL, snapshot_json TEXT NOT NULL)")
        retain_evidence(conn, "preparation:" + request["preparation_id"], _retention_entries({"snapshot": snapshot}))
        conn.execute("INSERT INTO cron_recovery_preparations VALUES (?,?,?)", (request["preparation_id"], json.dumps(request, sort_keys=True), json.dumps(snapshot, sort_keys=True)))
        return snapshot


def inspect_prepared_recovery_snapshot(request):
    """Explicit read-only absence; unreadable or drifted evidence is an error."""
    conn = sqlite3.connect(f"file:{incidents._db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN")
        return _prepared_snapshot(conn, request)
    finally:
        conn.rollback(); conn.close()


def _verify_authority(request, conn=None, *, current=True):
    authority = request.get("authority", {})
    documents = {}
    for name in ("baseline", "review"):
        proof = authority.get(name, {})
        content = Path(proof["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != proof.get("sha256"):
            raise ValueError("baseline authority artifact changed")
        documents[name] = json.loads(content)
    baseline, review = documents["baseline"], documents["review"]
    common = (
        baseline.get("mode") == "prospective_recovery_baseline_v1"
        and all(re.fullmatch(r"[0-9a-f]{40}", str(baseline.get(key, ""))) for key in ("reviewed_tezoff_sha", "reviewed_hermes_sha"))
        and baseline.get("authority_scope") in (None, "operational_health_only")
        and baseline.get("incident_disposition") == "preserved_unresolved"
        and baseline.get("cron_snapshot") == request.get("snapshot")
        and review.get("recovery_baseline_sha256") == authority["baseline"]["sha256"]
        and review.get("incident_disposition") == "preserved_unresolved"
        and review.get("verdict") == "pass"
        and review.get("reviewer_model") == "gpt-6-astra"
        and all(review.get(key) == baseline.get(key) for key in ("reviewed_tezoff_sha", "reviewed_hermes_sha"))
    )
    legacy = (review.get("schema_version") == 5 and review.get("review_version") == 5
              and review.get("replacement_authorized") is True
              and all(review.get("controls", {}).get(f"A{i}") == "pass" for i in range(1, 13)))
    health_only = (
        review.get("schema_version") == 1
        and review.get("mode") == "operational_incident_disposition_review_v1"
        and baseline.get("authority_scope") == "operational_health_only"
        and baseline.get("evidence_preparation") is not None
        and review.get("authority_scope") == "operational_health_only"
        and review.get("execution_authorized") is False
        and review.get("replacement_authorized") is False
        and review.get("collectors_activated") is False
        and review.get("financial_work_resolved") is False
        and all(review.get("controls", {}).get(f"H{i}") == "pass" for i in range(1, 6))
    )
    if not common or not (health_only if baseline.get("authority_scope") == "operational_health_only" else legacy):
        raise ValueError("baseline lacks exact prospective authority")
    if request.get("snapshot", {}).get("job_ids") is not None and not health_only:
        raise ValueError("scoped evidence cannot authorize replay")
    if baseline.get("evidence_preparation") is not None:
        if conn is None or _prepared_snapshot(conn, baseline["evidence_preparation"], current=current) != request["snapshot"]:
            raise ValueError("baseline lacks its exact prepared evidence")
    return baseline, review


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
        baseline, _ = _verify_authority(request, conn)
        if baseline.get("authority_succession") is not None:
            raise ValueError("a successor must use explicit baseline succession")
        if _snapshot(conn, job_id, job_ids=request["snapshot"].get("job_ids")) != request["snapshot"]:
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


def _stored_entry(conn, row):
    request = json.loads(row["request_json"])
    receipt = json.loads(row["receipt_json"])
    if (receipt.get("schema_version") != 1 or receipt.get("incident_disposition") != "preserved_unresolved"
            or receipt.get("baseline_id") != row["id"] or request.get("baseline_id") != row["id"]
            or row["id"] != _digest({"snapshot": request.get("snapshot"), "authority": request.get("authority")})
            or row["job_id"] != request.get("snapshot", {}).get("job_id")
            or receipt.get("request_sha256") != _digest(request)):
        raise ValueError("stored baseline receipt changed")
    _verify_authority(request, conn, current=False)
    _verify_retention(conn, request)
    return {"request": request, "receipt": receipt}


def _preserved_snapshot(previous, current):
    if previous.get("job_id") != current.get("job_id"):
        raise ValueError("baseline succession changed job identity")
    previous_scope = _job_scope(previous["job_id"], previous.get("job_ids"))
    current_scope = _job_scope(current["job_id"], current.get("job_ids"))
    if not set(previous_scope).issubset(current_scope):
        raise ValueError("baseline succession narrowed operational job scope")
    for field, key in (("incidents", "id"), ("failed_executions", "id"), ("logs", "incident_id")):
        rows = current[field]
        inventory = {row[key]: row for row in rows}
        if len(inventory) != len(rows) or any(inventory.get(row[key]) != row for row in previous[field]):
            raise ValueError("baseline succession changed preserved " + field)


def _succession_binding(baseline, review, previous):
    binding = baseline.get("authority_succession")
    expected = {
        "schema_version": 1,
        "mode": "reviewed_pre_activation_policy_succession_v1",
        "previous_baseline_id": previous["request"]["baseline_id"],
        "previous_request_sha256": _digest(previous["request"]),
        "previous_receipt_sha256": _digest(previous["receipt"]),
    }
    if (not isinstance(binding, dict) or set(binding) != set(expected) | {"maintenance_proof_sha256"}
            or any(binding.get(key) != value for key, value in expected.items())
            or not re.fullmatch(r"[0-9a-f]{64}", str(binding.get("maintenance_proof_sha256", "")))
            or review.get("authority_succession_sha256") != _digest(binding)
            or baseline.get("evidence_preparation") is None):
        raise ValueError("baseline succession lacks exact independently reviewed predecessor")
    return binding


def _baseline_chain(conn, root):
    chain = [_stored_entry(conn, root)]
    baseline, _ = _verify_authority(chain[0]["request"], conn, current=False)
    if baseline.get("authority_succession") is not None:
        raise ValueError("baseline root cannot be a successor")
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cron_recovery_successions'").fetchone():
        return chain
    for generation, row in enumerate(conn.execute(
            "SELECT * FROM cron_recovery_successions WHERE job_id=? ORDER BY generation", (root["job_id"],)), start=2):
        previous = chain[-1]
        if row["generation"] != generation or row["previous_id"] != previous["request"]["baseline_id"]:
            raise ValueError("baseline succession chain is not contiguous")
        entry = _stored_entry(conn, row)
        baseline, review = _verify_authority(entry["request"], conn, current=False)
        binding = _succession_binding(baseline, review, previous)
        if (entry["receipt"].get("authority_succession_sha256") != _digest(binding)
                or entry["receipt"].get("generation") != generation):
            raise ValueError("baseline succession receipt changed")
        _preserved_snapshot(previous["request"]["snapshot"], entry["request"]["snapshot"])
        chain.append(entry)
    return chain


def supersede_recovery_baseline(request):
    """Explicit reviewed CAS, never automatic recovery or incident closure.

    Keep the original registry row and every successor immutable. A fresh
    preparation and review may add failure evidence, but cannot remove or
    rewrite prior evidence. The caller must separately prove its unstarted
    maintenance boundary; this receipt grants no worker or replay capability.
    """
    if request.get("schema_version") != 1 or request.get("baseline_id") != _digest({
            "snapshot": request.get("snapshot"), "authority": request.get("authority")}):
        raise ValueError("baseline request identity changed")
    job_id = request["snapshot"]["job_id"]
    with incidents._transaction() as conn:
        conn.commit(); conn.execute("BEGIN IMMEDIATE")
        root = conn.execute("SELECT * FROM cron_recovery_baselines WHERE job_id=?", (job_id,)).fetchone()
        if root is None:
            raise ValueError("baseline succession predecessor is absent")
        chain = _baseline_chain(conn, root)
        current = chain[-1]
        baseline, review = _verify_authority(request, conn)
        if _snapshot(conn, job_id, job_ids=request["snapshot"].get("job_ids")) != request["snapshot"]:
            raise ValueError("baseline succession live snapshot changed")
        # A lost response can only adopt this exact already-current successor.
        if current["request"]["baseline_id"] == request["baseline_id"]:
            if len(chain) < 2 or current["request"] != request:
                raise ValueError("baseline succession retry differs")
            return current["receipt"]
        binding = _succession_binding(baseline, review, current)
        _preserved_snapshot(current["request"]["snapshot"], request["snapshot"])
        generation = len(chain) + 1
        receipt = {"schema_version": 1, "baseline_id": request["baseline_id"],
                   "incident_disposition": "preserved_unresolved", "request_sha256": _digest(request),
                   "registered_at": incidents._hermes_now().isoformat(), "generation": generation,
                   "authority_succession_sha256": _digest(binding)}
        conn.execute("CREATE TABLE IF NOT EXISTS cron_recovery_successions (id TEXT PRIMARY KEY, job_id TEXT NOT NULL, generation INTEGER NOT NULL, previous_id TEXT NOT NULL UNIQUE, request_json TEXT NOT NULL, receipt_json TEXT NOT NULL, UNIQUE(job_id,generation))")
        retain_evidence(conn, request["baseline_id"], _retention_entries(request))
        conn.execute("INSERT INTO cron_recovery_successions VALUES (?,?,?,?,?,?)", (
            request["baseline_id"], job_id, generation, current["request"]["baseline_id"],
            json.dumps(request, sort_keys=True), json.dumps(receipt, sort_keys=True)))
        return receipt


def inspect_recovery_authority_history(job_id):
    """Read preserved authority and new evidence without approving a change.

    Unlike current health, planning may observe additional failures. Every old
    row, log, artifact and retention pin must still match; uncertainty refuses.
    """
    job_ids = None
    if isinstance(job_id, dict):
        if set(job_id) != {"job_id", "job_ids"}:
            raise ValueError("unknown operational history selector")
        job_id, job_ids = job_id["job_id"], job_id["job_ids"]
    conn = sqlite3.connect(f"file:{incidents._db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN")
        root = conn.execute("SELECT * FROM cron_recovery_baselines WHERE job_id=?", (job_id,)).fetchone()
        if root is None:
            raise ValueError("baseline succession predecessor is absent")
        chain = _baseline_chain(conn, root)
        observed = _snapshot(conn, job_id, require_drained=False, job_ids=job_ids)
        _preserved_snapshot(chain[-1]["request"]["snapshot"], observed)
        return {"authority_history": chain, "current_snapshot": observed, "execution_authorized": False}
    finally:
        conn.rollback(); conn.close()


def inspect_recovery_baselines():
    """Read-only. A drifted checkpoint is an error, never an absent record."""
    conn = sqlite3.connect(f"file:{incidents._db_path()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN")
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cron_recovery_baselines'").fetchone():
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cron_recovery_successions'").fetchone():
                raise ValueError("baseline succession roots are missing")
            return []
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cron_recovery_successions'").fetchone():
            if conn.execute("SELECT 1 FROM cron_recovery_successions s LEFT JOIN cron_recovery_baselines b ON b.job_id=s.job_id WHERE b.id IS NULL LIMIT 1").fetchone():
                raise ValueError("baseline succession has an orphaned authority")
        output = []
        for row in conn.execute("SELECT * FROM cron_recovery_baselines ORDER BY id"):
            entry = _baseline_chain(conn, row)[-1]
            request = entry["request"]
            _verify_authority(request, conn)
            if _snapshot(conn, row["job_id"], require_drained=False, job_ids=request["snapshot"].get("job_ids")) != request["snapshot"]:
                raise ValueError("preserved incident or a later execution changed")
            output.append(entry)
        return output
    finally:
        conn.rollback(); conn.close()
